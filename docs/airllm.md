# Running open-weight models locally with AirLLM

[AirLLM](https://github.com/lyogavin/airllm) runs large open-weight models on small
hardware by streaming the model **one layer at a time** from disk instead of holding
it all in VRAM. A 70B model needs roughly 4 GB of VRAM this way rather than 140 GB.

The app talks to AirLLM through a small sidecar process
([`backend/airllm_server.py`](../backend/airllm_server.py)) that exposes the same
OpenAI-compatible API used for vLLM and Ollama.

---

## 1. Read this before you install

The layer-streaming trick trades memory for **time**: every token re-reads the whole
model from disk. That is a good trade on a GPU and a poor one on a CPU.

Measured on the development machine (Intel Core Ultra 9 185H, **no CUDA**, NVMe):

| Model | Throughput | One 512-token answer |
|---|---|---|
| Qwen2.5 **0.5B**, CPU, fp32 | **~10.4 s/token** | ~90 minutes |
| Qwen2.5 0.5B, CPU, bf16 | ~10.4 s/token (no change — it is I/O-bound, not dtype-bound) | ~90 minutes |

That is not a typo, and it is not specific to 0.5B: the cost is dominated by reading
~1 GB of layer shards per token, so it scales with model size, not with cleverness.

**On a CPU-only machine, use the Ollama provider instead.** It runs the same class of
open-weight models in seconds, and it is a first-class provider in the sidebar. AirLLM
earns its place when you have a GPU and want a model that does not fit in its VRAM.

### Which local backend should I use?

| Situation | Use |
|---|---|
| Model **fits** in your VRAM | **vLLM** (`medgemma` provider) — fastest |
| Model is **larger** than your VRAM | **AirLLM** — this is its design point |
| You want it to just work, quickly | **Ollama** — quantized GGUF, seconds per response |
| No GPU at all | **Ollama**. AirLLM only as a "I really want to run a 70B" experiment |

---

## 2. Install

### Step 1 — install torch for *your* hardware (do this first)

`backend/requirements-local.txt` deliberately does not list torch, because a wrong
wheel is the most common way this fails.

```bash
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GTX 10-series, RTX 20/30/40, most datacenter cards
pip install torch --index-url https://download.pytorch.org/whl/cu124

# RTX 50-series (Blackwell, sm_120) — needs torch >= 2.7 built against CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verify that your card's architecture is actually in the build:

```bash
python -c "import torch;print(torch.version.cuda, torch.cuda.get_arch_list())"
```

The list must contain your card's `sm_XX`. If it does not, the GPU will be detected
correctly and then fail at the first kernel launch with *"no kernel image is available
for execution on the device"*. This bites at **both** ends of the range: `sm_120`
(RTX 50) is missing from older wheels, and `sm_61` (GTX 10) has been dropped from
recent ones.

You do not have to remember this — the sidecar checks it at startup, logs an
actionable warning, and the app shows it in the sidebar.

### Step 2 — install AirLLM

```bash
pip install -r backend/requirements-local.txt
python -c "from airllm import AutoModel; print('ok')"
```

If that import fails after some future upgrade, isolate the sidecar rather than
downgrading the app — nothing else needs to change, because they talk over HTTP:

```bash
python -m venv .venv-airllm
.venv-airllm/bin/pip install airllm accelerate "transformers>=4.49,<5.0"
.venv-airllm/bin/python backend/airllm_server.py
```

---

## 3. Start the sidecar

```bash
# Windows
start.bat --airllm

# Linux / macOS
./start.sh --airllm

# or directly
python backend/airllm_server.py
```

It listens on **8002** (8000 = app, 8001 = vLLM, 11434 = Ollama).

```bash
curl http://localhost:8002/health
```

`/health` reports the detected hardware, the resolved dtype, and load progress as
`layers_split / layers_total`. The first load of a model downloads it and re-splits it
layer-by-layer, which takes minutes — this endpoint is how you watch that happen.

### Configuration

Every default below is derived from detected hardware
([`backend/hardware.py`](../backend/hardware.py)); set a variable only to override.

| Variable | Default | Purpose |
|---|---|---|
| `AIRLLM_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Model loaded at startup |
| `AIRLLM_MODELS` | — | Extra comma-separated ids to advertise in `/v1/models` |
| `AIRLLM_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` \| `cuda:1` \| `mps` |
| `AIRLLM_DTYPE` | bf16 on Ampere+, fp16 on Volta/Turing, **fp32 on Pascal and CPU** | Pascal runs fp16 at 1/64 rate; CPU has no fp16 matmul at all |
| `AIRLLM_MAX_SEQ_LEN` | 4096 CPU · 4096 (<8 GB) · 8192 (<16 GB) · 16384 (16 GB+) | KV cache is preallocated to this |
| `AIRLLM_MAX_NEW_TOKENS_CAP` | 512 CPU · 2048 CUDA | Server-side ceiling on output tokens |
| `AIRLLM_NUM_THREADS` | physical core count | CPU only; all logical threads is slower on hybrid CPUs |
| `AIRLLM_COMPRESSION` | — | `4bit` / `8bit`. Needs bitsandbytes; refused with an explanation where unsupported |
| `AIRLLM_SHARDS_DIR` | `backend/.airllm_shards` | Put this on a fast disk with room to spare |
| `AIRLLM_DELETE_ORIGINAL` | `0` | Delete the original HF snapshot after splitting, halving disk use |
| `AIRLLM_PRELOAD` | `0` | Load at startup instead of on first request |
| `AIRLLM_ALLOW_SWITCH` | `1` | Allow evicting the resident model to load another |
| `AIRLLM_PORT` / `AIRLLM_HOST` | `8002` / `127.0.0.1` | |
| `HF_TOKEN` | — | Required for gated repos (Llama) |

## 4. Point the app at it

The sidebar has a provider **and** a model dropdown per pipeline step, so you can mix:
extraction on a local model, the explanation on Claude. To set process-wide defaults:

```powershell
$env:LLM_PROVIDER_EXTRACT = "airllm"
$env:LLM_MODEL_EXTRACT    = "Qwen/Qwen2.5-7B-Instruct"
$env:LLM_PROVIDER_EXPLAIN = "claude"
python backend/app.py
```

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER_EXTRACT` / `_SELECT` / `_EXPLAIN` | `claude` \| `medgemma` \| `airllm` \| `ollama` |
| `LLM_MODEL_EXTRACT` / `_SELECT` / `_EXPLAIN` | Model id for that step |
| `AIRLLM_BASE_URL` | Default `http://127.0.0.1:8002/v1` |
| `LLM_ALLOW_UNLISTED_MODELS` | Accept a model id the catalog does not know about |

Use `127.0.0.1`, not `localhost`, in these URLs on Windows: `localhost` resolves to
IPv6 `::1` first and local servers usually bind IPv4 only, costing a full timeout on
every request (measured at 2.02 s vs 0.01 s).

## 5. Models

Disk cost is roughly **2x** the model size, because AirLLM keeps both the original
snapshot and the layer shards unless you set `AIRLLM_DELETE_ORIGINAL=1`.

| Model | Disk | Notes |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | ~2 GB | Smoke test. Verifies the setup end to end in minutes |
| `Qwen/Qwen2.5-1.5B-Instruct` | ~6 GB | Smallest that produces usable JSON |
| `Qwen/Qwen2.5-7B-Instruct` | ~30 GB | Good general default on a GPU |
| `mistralai/Mistral-7B-Instruct-v0.3` | ~30 GB | |
| `meta-llama/Llama-3.1-8B-Instruct` | ~32 GB | Gated — needs `HF_TOKEN` |
| `Qwen/Qwen2.5-32B-Instruct` | ~130 GB | Strong quality |
| `meta-llama/Llama-3.3-70B-Instruct` | ~280 GB | AirLLM's headline case: runs in ~4 GB VRAM |

## 6. Troubleshooting

**`no kernel image is available for execution on the device`** — your torch wheel has
no kernels for your GPU. See §2 step 1; check `torch.cuda.get_arch_list()`.

**`addmm_impl_cpu_ not implemented for 'Half'`** — fp16 on CPU. Unset `AIRLLM_DTYPE`
and let detection pick fp32.

**`OMP: Error #15 ... libiomp5md.dll already initialized`** — conda's duplicate OpenMP
runtime. The sidecar and both launchers set `KMP_DUPLICATE_LIB_OK=TRUE`; set it
yourself if you start Python another way.

**CUDA out of memory** — the sidecar returns HTTP 507 naming the model and your VRAM.
Lower `AIRLLM_MAX_SEQ_LEN` first; the KV cache is preallocated from it.

**Everything is extremely slow** — expected on CPU (§1). Switch to Ollama.

**Model never finishes loading** — watch `layers_split` in `/health`. If it is not
moving, check disk space in `AIRLLM_SHARDS_DIR`.

**Gated repo 401** — set `HF_TOKEN` and accept the licence on the model's HF page.

**Windows: long-path or symlink errors in the HF cache** — enable Developer Mode, or
point `HF_HOME` at a short path such as `C:\hf`.

**"The selected model did not return valid JSON"** in the chat — a small model failing
the extract/select steps. The parser tolerates fences, prose and trailing commas, but
it cannot invent structure. Use a larger model for those two steps; the explanation
step is far more forgiving.
