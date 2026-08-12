"""Per-judge GPU allocation for the runner notebooks.

Both judges now run unquantized at bf16, so neither fits on a single card:

    Qwen3-VL-32B-Thinking   ~64 GB of weights
    InternVL3.5-38B         ~76 GB of weights

Each A6000 exposes 47.4 GiB usable, so each judge is sharded across 3 cards by
Accelerate (`device_map="auto"`).

Placement matters. `nvidia-smi topo -m` on this box reports two NUMA islands:

    GPU 0 1 2 3   <- PIX (same PCIe switch, fast peer-to-peer)
    GPU 4 5 6 7   <- PIX
    across islands: SYS (hop via the CPU interconnect, slower)

Pipeline-parallel inference hands activations from one card to the next at every
layer-group boundary, so a model must stay inside one island. Keeping the two judges
in *different* islands also lets them run concurrently without competing for the same
PCIe switch - wall-clock for both is then the slower of the two, not their sum.

Usage in a runner notebook, in the FIRST cell, before torch initializes CUDA:

    from gpu_allocation import cuda_visible_devices_for, max_memory_for
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices_for("qwenvl")

then when constructing the judge:

    judge = JudgeModel("qwenvl", max_memory=max_memory_for("qwenvl"))

IMPORTANT: `CUDA_VISIBLE_DEVICES` renumbers devices. After setting it to "4,5,6",
torch sees exactly three devices numbered 0,1,2 - which is why `max_memory_for()`
returns keys 0..N-1 rather than the physical ids.
"""
import os

# Physical GPU ids per judge, chosen to keep each model inside one NUMA island.
JUDGE_GPUS = {
    "qwenvl": [4, 5, 6],     # island A
    "internvl": [0, 1, 2],   # island B
}

# Per-card ceiling handed to Accelerate. Cards are 47.4 GiB; 42 GiB leaves ~5 GiB for
# activations and KV cache. Worth setting explicitly: device_map="auto" alone tends to
# pack the first card tightly enough to OOM mid-generation - which, on a multi-hour run,
# surfaces hours in. Measured peaks on this workload were ~22 GiB/card for Qwen at bf16.
PER_CARD_MEMORY = "42GiB"


def cuda_visible_devices_for(*model_keys: str) -> str:
    """CUDA_VISIBLE_DEVICES string for one or more judges, e.g. "0,1,2".

    Pass every judge that will be loaded in this kernel. Loading two judges while only
    one judge's cards are visible would silently pack both into the same 3 cards and
    OOM (or spill to CPU, which is far worse - it turns hours into days)."""
    unknown = [k for k in model_keys if k not in JUDGE_GPUS]
    if unknown:
        raise ValueError(f"No GPU allocation for {unknown}; known judges: {list(JUDGE_GPUS)}")
    gpus = []
    for key in model_keys:
        gpus.extend(JUDGE_GPUS[key])
    return ",".join(str(g) for g in gpus)


def max_memory_for(model_key: str, model_keys_visible: tuple | list | None = None) -> dict:
    """`max_memory` for JudgeModel, keyed by the LOGICAL device ids torch sees after
    CUDA_VISIBLE_DEVICES has been applied.

    `model_keys_visible` must list the judges in the same order passed to
    `cuda_visible_devices_for()`. With one judge visible it can be omitted."""
    if model_key not in JUDGE_GPUS:
        raise ValueError(f"No GPU allocation for {model_key!r}; known judges: {list(JUDGE_GPUS)}")
    if model_keys_visible is None:
        model_keys_visible = (model_key,)
    if model_key not in model_keys_visible:
        raise ValueError(f"{model_key!r} is not among the visible judges {list(model_keys_visible)}")

    # Walk the visible judges in order, assigning consecutive logical ids.
    logical = 0
    for key in model_keys_visible:
        n = len(JUDGE_GPUS[key])
        if key == model_key:
            return {logical + i: PER_CARD_MEMORY for i in range(n)}
        logical += n
    raise AssertionError("unreachable")


def describe_allocation() -> str:
    lines = [f"PER_CARD_MEMORY = {PER_CARD_MEMORY}"]
    for key, gpus in JUDGE_GPUS.items():
        island = "A (0-3)" if all(g < 4 for g in gpus) else "B (4-7)"
        lines.append(f"  {key:9s} -> physical GPUs {gpus}  [NUMA island {island}]")
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    lines.append(f"  CUDA_VISIBLE_DEVICES currently = {env!r}")
    return "\n".join(lines)
