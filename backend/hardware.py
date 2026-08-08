"""
Hardware detection and device policy for local inference.

Single source of truth for device, dtype, sequence length, token caps, and client
timeouts, so nothing downstream hardcodes an assumption about the machine it runs
on. The project targets everything from a CPU-only laptop through GTX 10-series
(Pascal), RTX 20/30/40, and RTX 50-series (Blackwell) up to datacenter cards.

This matters because AirLLM provides no fallback logic at all: its `device`
argument defaults to the literal string "cuda:0" and its `dtype` defaults to
float16, both of which are wrong on several of the tiers above.

Importable WITHOUT torch: `profile()` returns an "unknown" profile in that case,
so the web app (which has no torch dependency) can still call `describe()`.

Consumers:
- `backend/airllm_server.py` — device/dtype/max_seq_len/threads for model loading
- `backend/llm_providers.py`  — client timeout and token cap for the AirLLM provider
"""
import logging
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple

# Conda environments commonly link two OpenMP runtimes, and torch aborts the whole
# process on import with "OMP: Error #15 ... libiomp5md.dll already initialized".
# torch is imported lazily below, so setting this here covers every entry point
# (app, sidecar, tests) without each one having to remember.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger(__name__)


# ── CUDA architecture families ───────────────────────────────────────────────
# (major, minor) -> (family label, marketing examples)
_FAMILIES = {
    (5, 0): ("Maxwell", "GTX 750/900 series"),
    (5, 2): ("Maxwell", "GTX 900 series"),
    (6, 0): ("Pascal", "Tesla P100"),
    (6, 1): ("Pascal", "GTX 10-series / Titan Xp"),
    (7, 0): ("Volta", "Tesla V100 / Titan V"),
    (7, 5): ("Turing", "RTX 20-series / GTX 16-series"),
    (8, 0): ("Ampere", "A100"),
    (8, 6): ("Ampere", "RTX 30-series"),
    (8, 7): ("Ampere", "Jetson Orin"),
    (8, 9): ("Ada Lovelace", "RTX 40-series"),
    (9, 0): ("Hopper", "H100 / H200"),
    (10, 0): ("Blackwell", "B100 / B200 (datacenter)"),
    (10, 3): ("Blackwell", "GB200 (datacenter)"),
    (12, 0): ("Blackwell", "RTX 50-series"),
}

#: First capability with usable bf16. Below this, fp16 or fp32.
_BF16_MIN = (8, 0)
#: First capability with real fp16 throughput. Consumer Pascal (6.1) runs fp16 at
#: 1/64 rate, so fp16 there is far SLOWER than fp32 despite halving memory.
_FP16_MIN = (7, 0)
#: bitsandbytes int8/nf4 kernels require Turing or newer...
_BNB_MIN = (7, 5)
#: ...and have no working Blackwell (sm_120 / sm_100) support yet.
_BNB_MAX_EXCLUSIVE = (10, 0)

#: Architectures that need CUDA 12.8+ toolkits (PyTorch 2.7+ cu128 wheels).
_NEEDS_CU128 = ((10, 0), (10, 3), (12, 0))

_CU128_HINT = "pip install torch --index-url https://download.pytorch.org/whl/cu128"
_CU124_HINT = "pip install torch --index-url https://download.pytorch.org/whl/cu124"


def family_of(capability: Tuple[int, ...]) -> str:
    """Human architecture name for a compute capability, with a forward-compatible guess."""
    if not capability:
        return "unknown"
    exact = _FAMILIES.get(tuple(capability))
    if exact:
        return exact[0]
    major = capability[0]
    # Unknown minor within a known major (e.g. a future sm_89-like part).
    for (maj, _minor), (label, _ex) in _FAMILIES.items():
        if maj == major:
            return label
    return f"sm_{major}{capability[1]}"


@dataclass(frozen=True)
class HardwareProfile:
    backend: str = "unknown"            # "cuda" | "cpu" | "mps" | "unknown"
    device: str = "cpu"                 # "cuda:0" | "cpu" | "mps"
    name: str = "unknown"
    family: str = "unknown"             # "Pascal" | "Turing" | "Ampere" | "Blackwell" | ...
    vram_gb: float = 0.0
    capability: Tuple[int, ...] = ()
    torch_version: Optional[str] = None
    torch_cuda_build: Optional[str] = None
    arch_list: Tuple[str, ...] = ()
    dtype: str = "float32"
    tier: str = "cpu"                   # cpu | cuda_tiny | cuda_small | cuda_large | mps | unknown
    supports_bnb: bool = False
    cpu_threads: int = 0
    max_seq_len: int = 4096
    max_new_tokens_cap: int = 512
    client_timeout: float = 3600.0
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_cuda(self) -> bool:
        return self.backend == "cuda"


# ── env helpers ──────────────────────────────────────────────────────────────

def _env(name: str) -> Optional[str]:
    val = os.environ.get(name, "").strip()
    return val or None


def _env_int(name: str) -> Optional[int]:
    val = _env(name)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r", name, val)
        return None


def _physical_cores() -> int:
    """Physical core count, falling back to a sane guess.

    Caps torch's CPU thread pool. On hybrid CPUs (e.g. Intel Core Ultra with
    6 P-cores + 16 E-cores) letting torch grab every logical thread bounces work
    onto E-cores and is measurably slower than using fewer.
    """
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
        if cores:
            return int(cores)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 4) // 2)


def _cpu_name() -> str:
    import platform
    if platform.system() == "Windows":
        return os.environ.get("PROCESSOR_IDENTIFIER", "") or platform.processor() or "cpu"
    return platform.processor() or platform.machine() or "cpu"


# ── policy ───────────────────────────────────────────────────────────────────

def _resolve_dtype(backend: str, capability: Tuple[int, ...], torch_mod) -> str:
    """Pick a dtype that is both implemented and actually fast on this device."""
    override = _env("AIRLLM_DTYPE")
    if override:
        return override

    if backend == "cuda":
        if capability >= _BF16_MIN:
            return "bfloat16"           # Ampere (RTX 30) and newer
        if capability >= _FP16_MIN:
            return "float16"            # Volta / Turing (RTX 20, GTX 16)
        return "float32"                # Pascal (GTX 10) and older: fp16 is 1/64 rate

    if backend == "mps":
        return "float16"

    # CPU: fp16 matmul is unimplemented ("addmm_impl_cpu_ not implemented for
    # 'Half'"); bf16 is only worth it with AVX512-BF16 hardware support.
    try:
        if hasattr(torch_mod.cpu, "_is_avx512_bf16_supported") and torch_mod.cpu._is_avx512_bf16_supported():
            return "bfloat16"
    except Exception:
        pass
    return "float32"


def _arch_supported(capability: Tuple[int, ...], arch_list: Tuple[str, ...]) -> bool:
    """Does this torch build actually ship kernels for this card?

    Generalizes what is usually described as "the Blackwell problem". It cuts both
    ways: new cards (sm_120) need a wheel newer than the toolkit that built the
    common one, and OLD cards (sm_61 Pascal) get dropped from recent wheels. Both
    present identically at runtime as "no kernel image is available for execution
    on the device", thrown at the first kernel launch rather than at load.

    PTX forward-compat: a build with `compute_XY` PTX for an OLDER arch can JIT for
    a newer one of the same major, so treat a compute_ entry <= this capability as
    covering it.
    """
    if not capability or not arch_list:
        return True  # nothing to check against; don't cry wolf

    target = capability[0] * 10 + capability[1]
    for entry in arch_list:
        digits = "".join(ch for ch in entry if ch.isdigit())
        if not digits:
            continue
        try:
            value = int(digits)
        except ValueError:
            continue
        if value == target:
            return True
        # PTX can JIT forward within the same major version.
        if entry.startswith("compute_") and value <= target and value // 10 == capability[0]:
            return True
    return False


def _arch_warning(name: str, capability, family: str, arch_list, torch_version, torch_cuda_build) -> str:
    sm = f"sm_{capability[0]}{capability[1]}"
    hint = _CU128_HINT if tuple(capability) in _NEEDS_CU128 else _CU124_HINT
    base = (
        f"{name} is {family} ({sm}) but torch {torch_version} (CUDA {torch_cuda_build}) was built for "
        f"{list(arch_list)} — it has no {sm} kernels and will fail at the first kernel launch with "
        f"'no kernel image is available for execution on the device'. "
    )
    if capability < (7, 0):
        return base + (
            f"Recent PyTorch wheels have dropped {family} support; either install an older torch "
            f"(2.4-2.6 line) or use the Ollama provider, which ships its own kernels."
        )
    return base + f"Install a matching CUDA build: {hint}"


def _cuda_tier(vram_gb: float) -> Tuple[str, int, int]:
    """(tier, default max_seq_len, default max_new_tokens cap) by VRAM.

    AirLLM streams weights layer-by-layer, so VRAM barely constrains model SIZE —
    but the KV cache is preallocated to max_seq_len, and that does scale with VRAM.
    A 4-6 GB Pascal/Turing card will OOM on the same setting a 24 GB card shrugs at.
    """
    if vram_gb < 8:
        return "cuda_tiny", 4096, 1024
    if vram_gb < 16:
        return "cuda_small", 8192, 2048
    return "cuda_large", 16384, 2048


def _detect() -> HardwareProfile:
    warnings = []

    try:
        import torch
    except ImportError:
        return HardwareProfile(
            backend="unknown",
            warnings=("torch is not installed — local inference via AirLLM is unavailable. "
                      "See backend/requirements-local.txt.",),
        )

    torch_version = getattr(torch, "__version__", None)
    torch_cuda_build = getattr(getattr(torch, "version", None), "cuda", None)

    requested = (_env("AIRLLM_DEVICE") or "auto").lower()
    cuda_ok = bool(getattr(torch, "cuda", None)) and torch.cuda.is_available()
    try:
        mps_ok = torch.backends.mps.is_available()
    except Exception:
        mps_ok = False

    if requested == "auto":
        backend = "cuda" if cuda_ok else ("mps" if mps_ok else "cpu")
        device = "cuda:0" if backend == "cuda" else backend
    elif requested.startswith("cuda"):
        backend, device = "cuda", (requested if ":" in requested else "cuda:0")
        if not cuda_ok:
            # The most confusing failure to debug: a +cpu wheel has no CUDA at all,
            # so name the wheel explicitly rather than saying "CUDA unavailable".
            warnings.append(
                f"AIRLLM_DEVICE={requested} was requested but torch {torch_version} reports no usable CUDA "
                f"(torch.version.cuda={torch_cuda_build!r} — this looks like a CPU-only build). "
                f"Install a CUDA build ({_CU128_HINT}) or unset AIRLLM_DEVICE to fall back to CPU."
            )
            backend, device = "cpu", "cpu"
    elif requested == "mps":
        if mps_ok:
            backend, device = "mps", "mps"
        else:
            warnings.append("AIRLLM_DEVICE=mps requested but MPS is unavailable; falling back to CPU.")
            backend, device = "cpu", "cpu"
    else:
        backend, device = "cpu", "cpu"

    name, family = "cpu", "unknown"
    vram_gb = 0.0
    capability: Tuple[int, ...] = ()
    arch_list: Tuple[str, ...] = ()

    if backend == "cuda":
        index = int(device.split(":")[1]) if ":" in device else 0
        count = torch.cuda.device_count()
        if index >= count:
            warnings.append(
                f"AIRLLM_DEVICE={device} but only {count} CUDA device(s) present; using cuda:0."
            )
            index, device = 0, "cuda:0"
        try:
            props = torch.cuda.get_device_properties(index)
            name = props.name
            vram_gb = round(props.total_memory / (1024 ** 3), 1)
            capability = (props.major, props.minor)
            family = family_of(capability)
        except Exception as exc:
            warnings.append(f"Could not query CUDA device {index}: {exc}")
        try:
            arch_list = tuple(torch.cuda.get_arch_list())
        except Exception:
            arch_list = ()

        if capability and not _arch_supported(capability, arch_list):
            warnings.append(
                _arch_warning(name, capability, family, arch_list, torch_version, torch_cuda_build)
            )
        if capability and capability < (6, 0):
            warnings.append(
                f"{family} ({name}) is older than PyTorch's supported range; local inference is "
                "unlikely to work. Consider the Ollama provider instead."
            )
    elif backend == "mps":
        name, family = "Apple Silicon", "Apple Silicon"

    dtype = _resolve_dtype(backend, capability, torch)

    # bitsandbytes is needed only for AIRLLM_COMPRESSION. Gate rather than crash:
    # it needs Turing+ for its int8/nf4 kernels and has no Blackwell support yet.
    import importlib.util
    has_bnb = importlib.util.find_spec("bitsandbytes") is not None
    supports_bnb = (
        has_bnb and backend == "cuda"
        and _BNB_MIN <= capability < _BNB_MAX_EXCLUSIVE
    )
    if has_bnb and backend == "cuda" and capability:
        if capability >= _BNB_MAX_EXCLUSIVE:
            warnings.append(
                f"bitsandbytes has no working {family} (sm_{capability[0]}{capability[1]}) support — "
                "4bit/8bit compression is disabled on this GPU. Run without AIRLLM_COMPRESSION."
            )
        elif capability < _BNB_MIN:
            warnings.append(
                f"bitsandbytes requires Turing (sm_75) or newer; {family} cannot use "
                "4bit/8bit compression. Run without AIRLLM_COMPRESSION."
            )

    if backend == "cuda":
        tier, default_seq, default_cap = _cuda_tier(vram_gb)
        # Pascal/Turing in fp32 move twice the bytes per layer, so budget more time.
        timeout = 900.0 if capability >= _BF16_MIN else 1800.0
        threads = 0
    elif backend == "mps":
        tier, default_seq, default_cap, timeout, threads = "mps", 8192, 1024, 1800.0, 0
    else:
        # CPU streams every layer from disk per forward pass. Cap output tokens so a
        # 2000-token explain step cannot run for hours.
        tier, default_seq, default_cap, timeout = "cpu", 4096, 512, 3600.0
        threads = _env_int("AIRLLM_NUM_THREADS") or _physical_cores()
        try:
            name = _cpu_name()
        except Exception:
            name = "cpu"

    return HardwareProfile(
        backend=backend,
        device=device,
        name=name,
        family=family,
        vram_gb=vram_gb,
        capability=capability,
        torch_version=torch_version,
        torch_cuda_build=torch_cuda_build,
        arch_list=arch_list,
        dtype=dtype,
        tier=tier,
        supports_bnb=supports_bnb,
        cpu_threads=threads,
        max_seq_len=_env_int("AIRLLM_MAX_SEQ_LEN") or default_seq,
        max_new_tokens_cap=_env_int("AIRLLM_MAX_NEW_TOKENS_CAP") or default_cap,
        client_timeout=float(_env("AIRLLM_CLIENT_TIMEOUT") or timeout),
        warnings=tuple(warnings),
    )


# ── public API ───────────────────────────────────────────────────────────────

_PROFILE: Optional[HardwareProfile] = None


def profile(refresh: bool = False) -> HardwareProfile:
    """Memoized hardware profile. Detection touches torch, so do it once."""
    global _PROFILE
    if _PROFILE is None or refresh:
        _PROFILE = _detect()
    return _PROFILE


def summary(p: Optional[HardwareProfile] = None) -> str:
    """One-line description, e.g. 'CUDA · NVIDIA GeForce RTX 3060 (Ampere) · 12.0 GB · bfloat16'."""
    p = p or profile()
    if p.backend == "cuda":
        return f"CUDA · {p.name} ({p.family}) · {p.vram_gb} GB · {p.dtype}"
    if p.backend == "mps":
        return f"Apple Silicon (MPS) · {p.dtype}"
    if p.backend == "cpu":
        return f"CPU · {p.name} · {p.cpu_threads} threads · {p.dtype}"
    return "unknown (torch not installed)"


def describe(refresh: bool = False) -> dict:
    """JSON-serializable profile for /health and /api/config."""
    p = profile(refresh=refresh)
    data = asdict(p)
    data["capability"] = ".".join(str(x) for x in p.capability) if p.capability else None
    data["arch_list"] = list(p.arch_list)
    data["warnings"] = list(p.warnings)
    data["summary"] = summary(p)
    return data


def torch_dtype():
    """The torch dtype object matching the resolved policy."""
    import torch
    return getattr(torch, profile().dtype)


if __name__ == "__main__":  # `python backend/hardware.py` prints the profile
    import json as _json
    print(_json.dumps(describe(), indent=2))
