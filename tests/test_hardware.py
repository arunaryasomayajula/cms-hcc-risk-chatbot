"""
Hardware-policy tests.

The point of these is coverage of GPU tiers nobody running them owns. `hardware`
imports torch lazily inside `_detect()`, so we can swap a fake torch into
sys.modules and drive every branch — Pascal through Blackwell — from a CPU laptop.

Run standalone:  python tests/test_hardware.py
Or with pytest:  pytest tests/test_hardware.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import hardware  # noqa: E402


# ── fake torch ───────────────────────────────────────────────────────────────

class _Props:
    def __init__(self, name, major, minor, total_memory):
        self.name = name
        self.major = major
        self.minor = minor
        self.total_memory = total_memory


def fake_torch(*, cuda_available=False, name="", capability=(0, 0), vram_gb=0,
               arch_list=(), cuda_build=None, version="2.12.0", avx512_bf16=False,
               device_count=1, mps=False):
    t = types.ModuleType("torch")
    t.__version__ = version

    t.version = types.SimpleNamespace(cuda=cuda_build)

    t.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: device_count,
        get_arch_list=lambda: list(arch_list),
        get_device_properties=lambda i: _Props(name, capability[0], capability[1],
                                               int(vram_gb * 1024 ** 3)),
    )
    t.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    t.cpu = types.SimpleNamespace(_is_avx512_bf16_supported=lambda: avx512_bf16)

    for dt in ("float32", "float16", "bfloat16"):
        setattr(t, dt, dt)
    return t


def detect_with(fake, **env):
    """Run detection against a fake torch and a controlled environment."""
    saved_torch = sys.modules.get("torch")
    saved_env = {k: os.environ.get(k) for k in (
        "AIRLLM_DEVICE", "AIRLLM_DTYPE", "AIRLLM_MAX_SEQ_LEN",
        "AIRLLM_MAX_NEW_TOKENS_CAP", "AIRLLM_NUM_THREADS", "AIRLLM_CLIENT_TIMEOUT",
    )}
    try:
        for k in saved_env:
            os.environ.pop(k, None)
        for k, v in env.items():
            os.environ[k] = str(v)
        sys.modules["torch"] = fake
        return hardware.profile(refresh=True)
    finally:
        for k, v in saved_env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        if saved_torch is not None:
            sys.modules["torch"] = saved_torch
        else:
            sys.modules.pop("torch", None)
        hardware.profile(refresh=True)


# ── presets ──────────────────────────────────────────────────────────────────

PASCAL = dict(cuda_available=True, name="NVIDIA GeForce GTX 1080 Ti", capability=(6, 1),
              vram_gb=11, cuda_build="12.4")
TURING = dict(cuda_available=True, name="NVIDIA GeForce RTX 2060", capability=(7, 5),
              vram_gb=6, cuda_build="12.4")
AMPERE = dict(cuda_available=True, name="NVIDIA GeForce RTX 3060", capability=(8, 6),
              vram_gb=12, cuda_build="12.4")
ADA = dict(cuda_available=True, name="NVIDIA GeForce RTX 4090", capability=(8, 9),
           vram_gb=24, cuda_build="12.4")
BLACKWELL = dict(cuda_available=True, name="NVIDIA GeForce RTX 5090", capability=(12, 0),
                 vram_gb=32, cuda_build="12.8")

ARCH_12X = ("sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120")
ARCH_124 = ("sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90")
ARCH_MODERN = ("sm_75", "sm_80", "sm_86", "sm_90")  # no sm_61: Pascal dropped


# ── tests ────────────────────────────────────────────────────────────────────

def test_cpu_tier():
    p = detect_with(fake_torch())
    assert p.backend == "cpu" and p.device == "cpu"
    assert p.dtype == "float32", "fp16 matmul is unimplemented on CPU"
    assert p.tier == "cpu"
    assert p.max_new_tokens_cap == 512, "CPU must cap output or explain runs for hours"
    assert p.client_timeout == 3600.0
    assert p.cpu_threads > 0
    assert not p.supports_bnb


def test_cpu_with_avx512_bf16():
    p = detect_with(fake_torch(avx512_bf16=True))
    assert p.dtype == "bfloat16"


def test_pascal_uses_fp32_not_fp16():
    # GTX 10-series runs fp16 at 1/64 rate; fp16 there is slower than fp32.
    p = detect_with(fake_torch(**PASCAL, arch_list=ARCH_124))
    assert p.family == "Pascal"
    assert p.dtype == "float32"
    assert p.tier == "cuda_small"          # 11 GB
    assert not p.supports_bnb, "bitsandbytes needs Turing+"


def test_turing_uses_fp16_and_tiny_tier():
    p = detect_with(fake_torch(**TURING, arch_list=ARCH_124))
    assert p.family == "Turing"
    assert p.dtype == "float16", "Turing has real fp16 but no bf16"
    assert p.tier == "cuda_tiny", "6 GB card must get a smaller KV budget"
    assert p.max_seq_len == 4096


def test_ampere_uses_bf16():
    p = detect_with(fake_torch(**AMPERE, arch_list=ARCH_124))
    assert p.family == "Ampere"
    assert p.dtype == "bfloat16"
    assert p.tier == "cuda_small"          # 12 GB
    assert p.max_new_tokens_cap == 2048, "GPU should not inherit the CPU cap"
    assert p.client_timeout == 900.0


def test_ada_large_tier():
    p = detect_with(fake_torch(**ADA, arch_list=ARCH_124))
    assert p.family == "Ada Lovelace"
    assert p.tier == "cuda_large" and p.max_seq_len == 16384


def test_blackwell_with_correct_wheel_has_no_warnings():
    p = detect_with(fake_torch(**BLACKWELL, arch_list=ARCH_12X))
    assert p.family == "Blackwell"
    assert p.dtype == "bfloat16"
    assert p.tier == "cuda_large"
    assert not p.warnings, f"unexpected warnings: {p.warnings}"


def test_blackwell_with_old_wheel_warns_with_install_line():
    """The signature Blackwell failure: card detected, kernels absent."""
    p = detect_with(fake_torch(**{**BLACKWELL, "cuda_build": "12.4"}, arch_list=ARCH_124))
    assert p.warnings, "must warn when sm_120 kernels are missing"
    joined = " ".join(p.warnings)
    assert "sm_120" in joined
    assert "cu128" in joined, "warning must carry the exact fix"
    assert "no kernel image" in joined


def test_pascal_dropped_from_modern_wheel_warns():
    """Same failure mode as Blackwell, opposite end: recent wheels drop sm_61."""
    p = detect_with(fake_torch(**PASCAL, arch_list=ARCH_MODERN))
    joined = " ".join(p.warnings)
    assert "sm_61" in joined
    assert "Ollama" in joined, "old cards should be pointed at a working alternative"


def test_ptx_forward_compat_is_not_flagged():
    """A compute_86 PTX entry can JIT for sm_89, so don't cry wolf."""
    p = detect_with(fake_torch(**ADA, arch_list=("sm_80", "sm_86", "compute_86")))
    assert not p.warnings, f"unexpected warnings: {p.warnings}"


def test_cuda_requested_but_cpu_wheel_installed():
    p = detect_with(fake_torch(), AIRLLM_DEVICE="cuda")
    assert p.backend == "cpu", "must fall back rather than crash"
    joined = " ".join(p.warnings)
    assert "CPU-only build" in joined
    assert "cu128" in joined


def test_device_index_out_of_range_falls_back():
    p = detect_with(fake_torch(**AMPERE, arch_list=ARCH_124, device_count=1),
                    AIRLLM_DEVICE="cuda:3")
    assert p.device == "cuda:0"
    assert any("only 1 CUDA device" in w for w in p.warnings)


def test_env_overrides_win():
    p = detect_with(fake_torch(**AMPERE, arch_list=ARCH_124),
                    AIRLLM_DTYPE="float16", AIRLLM_MAX_SEQ_LEN="2048",
                    AIRLLM_MAX_NEW_TOKENS_CAP="128")
    assert p.dtype == "float16"
    assert p.max_seq_len == 2048
    assert p.max_new_tokens_cap == 128


def test_mps():
    p = detect_with(fake_torch(mps=True))
    assert p.backend == "mps" and p.dtype == "float16"


def test_no_torch_is_not_fatal():
    saved = sys.modules.get("torch")
    try:
        sys.modules["torch"] = None  # forces ImportError in `import torch`
        p = hardware.profile(refresh=True)
        assert p.backend == "unknown"
        assert "torch is not installed" in " ".join(p.warnings)
        assert isinstance(hardware.describe(), dict)
    finally:
        if saved is not None:
            sys.modules["torch"] = saved
        else:
            sys.modules.pop("torch", None)
        hardware.profile(refresh=True)


def test_describe_is_json_serializable():
    import json
    p = detect_with(fake_torch(**BLACKWELL, arch_list=ARCH_12X))
    assert p.family == "Blackwell"
    json.dumps(hardware.describe())


if __name__ == "__main__":
    failures = 0
    for fname, fn in sorted(globals().items()):
        if fname.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {fname}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {fname}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  ERROR {fname}: {type(exc).__name__}: {exc}")
    print(f"\n{'FAILED' if failures else 'OK'} — {failures} failure(s)")
    sys.exit(1 if failures else 0)
