"""Which GPU each answerer runs on.

Unlike the judges - 32B/38B models that each needed THREE cards sharded by Accelerate -
every answerer here is 4-8B and fits on one card with room to spare. Measured resident
sizes:

    internvl8b  15.89 GiB      llavamed    14.59 GiB
    qwen25vl    15.45 GiB      quiltllava  13.22 GiB
    pathor1     15.45 GiB      medgemma     8.01 GiB

An A6000 exposes 47.4 GiB, so two 7B models co-resident is ~31 GiB - comfortable, with
~16 GiB left for activations and KV cache. That is what makes a single wave possible.

## Why co-residency is safe here but was not for the judges

The judges were sharded ACROSS cards, so two judges on one card would have contended for
the same PCIe switch on every layer-group boundary. These models are self-contained: a
co-resident pair shares only raw SM time, and since each process spends most of its wall
clock in the Python/tokenizer path rather than saturating the GPU, the two interleave
rather than queue. Expect some slowdown per model, but far less than running them in
two sequential waves.

## Placement

GPUs 0-2 are occupied by the Qwen judge's Benchmark 5 run in `../vlm/` and MUST NOT be
touched - that job has days of accumulated progress and its checkpoint path is absolute.
Only 3-7 are available.

    GPU 3   qwen25vl   + medgemma      15.45 + 8.01  = 23.5 GiB
    GPU 4   internvl8b + pathor1       15.89 + 15.45 = 31.3 GiB
    GPU 5   llavamed   + quiltllava    14.59 + 13.22 = 27.8 GiB   (path-llava env)
    GPU 6   spare - headroom, or move a pair here if 4 runs hot
    GPU 7   spare

The two LLaVA models are paired on one card deliberately: they run in a different conda
environment, so keeping them together means only one GPU hosts the second environment and
the other cards stay uniform.

`nvidia-smi topo -m` reports two NUMA islands (0-3 and 4-7). That mattered for the sharded
judges; it does not here, because no model spans cards. Pairs are chosen by memory fit
rather than topology.
"""
from __future__ import annotations

import os

# Physical GPU per model. Two models sharing an id run co-resident on that card.
MODEL_GPU = {
    "qwen25vl": 3,
    "medgemma": 3,
    "internvl8b": 4,
    "pathor1": 4,
    "llavamed": 5,
    "quiltllava": 5,
}

# Measured resident size, GiB. Used by `describe_allocation` to show headroom, and by
# `check_fit` to refuse an allocation that cannot work.
MODEL_MEMORY_GIB = {
    "qwen25vl": 15.45,
    "internvl8b": 15.89,
    "medgemma": 8.01,
    "llavamed": 14.59,
    "quiltllava": 13.22,
    "pathor1": 15.45,
}

# A6000 usable capacity. The card reports 49140 MiB total; ~47.4 GiB is addressable after
# driver overhead.
GPU_CAPACITY_GIB = 47.4
# Leave this much per card for activations, KV cache and fragmentation. Generation is
# short here (16-256 new tokens), so the working set is small - but a co-resident pair
# doubles the transient allocations, and an OOM hours into a run is far more expensive
# than being conservative now.
RESERVE_GIB = 10.0

# GPUs 0-2 belong to the judge run in ../vlm/. Listed explicitly so `check_fit` can refuse
# them rather than relying on everyone remembering.
RESERVED_GPUS = frozenset({0, 1, 2})

# Which conda environment each model needs. The runner uses this to pick the interpreter.
MODEL_ENV = {
    "qwen25vl": "main",
    "internvl8b": "main",
    "medgemma": "main",
    "pathor1": "main",
    "llavamed": "llava",
    "quiltllava": "llava",
}

PYTHON = {
    "main": "/data/mn27889/miniconda3/envs/path-opendata-vlms/bin/python",
    "llava": "/data/mn27889/miniconda3/envs/path-llava/bin/python",
}


def gpu_for(model_key: str) -> int:
    if model_key not in MODEL_GPU:
        raise ValueError(f"no GPU assigned to {model_key!r}; known: {list(MODEL_GPU)}")
    return MODEL_GPU[model_key]


def models_on(gpu: int) -> list:
    return sorted(k for k, g in MODEL_GPU.items() if g == gpu)


def env_for(model_key: str) -> str:
    return MODEL_ENV[model_key]


def python_for(model_key: str) -> str:
    return PYTHON[MODEL_ENV[model_key]]


def cuda_visible_devices_for(model_key: str) -> str:
    """Value for CUDA_VISIBLE_DEVICES. Set this BEFORE torch initializes CUDA.

    One physical id is exposed, so the model always addresses `cuda:0` regardless of which
    card it actually lands on - the runner never has to translate device indices."""
    return str(gpu_for(model_key))


def check_fit(model_keys=None) -> list:
    """Verify every card's co-resident models fit. Returns a list of problems (empty = OK).

    Called by the runner before any model loads: a bad allocation should fail in the first
    second, not after one model has spent a minute loading and the second OOMs."""
    if model_keys is None:
        model_keys = list(MODEL_GPU)
    problems = []
    by_gpu = {}
    for key in model_keys:
        by_gpu.setdefault(gpu_for(key), []).append(key)

    for gpu, keys in sorted(by_gpu.items()):
        if gpu in RESERVED_GPUS:
            problems.append(
                f"GPU {gpu} is reserved for the judge run in ../vlm/ but was assigned "
                f"{keys}; that job has days of progress and must not be disturbed")
        total = sum(MODEL_MEMORY_GIB.get(k, 16.0) for k in keys)
        budget = GPU_CAPACITY_GIB - RESERVE_GIB
        if total > budget:
            problems.append(
                f"GPU {gpu}: {keys} need {total:.1f} GiB but only {budget:.1f} GiB is "
                f"available after the {RESERVE_GIB:.0f} GiB activation reserve")
    return problems


def describe_allocation(model_keys=None) -> str:
    if model_keys is None:
        model_keys = list(MODEL_GPU)
    by_gpu = {}
    for key in model_keys:
        by_gpu.setdefault(gpu_for(key), []).append(key)

    lines = [f"capacity {GPU_CAPACITY_GIB} GiB/card, reserving {RESERVE_GIB} GiB for activations"]
    for gpu in sorted(by_gpu):
        keys = sorted(by_gpu[gpu])
        total = sum(MODEL_MEMORY_GIB.get(k, 16.0) for k in keys)
        envs = {MODEL_ENV[k] for k in keys}
        detail = " + ".join(f"{k} ({MODEL_MEMORY_GIB.get(k, 16.0):.1f})" for k in keys)
        lines.append(f"  GPU {gpu}: {detail} = {total:.1f} GiB   "
                     f"[{GPU_CAPACITY_GIB - total:.1f} GiB free, env: {'/'.join(sorted(envs))}]")
    for gpu in sorted(set(range(8)) - set(by_gpu) - RESERVED_GPUS):
        lines.append(f"  GPU {gpu}: free")
    for gpu in sorted(RESERVED_GPUS):
        lines.append(f"  GPU {gpu}: RESERVED (judge run in ../vlm/)")
    problems = check_fit(model_keys)
    lines.append("  allocation OK" if not problems else "  PROBLEMS: " + "; ".join(problems))
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_allocation())
