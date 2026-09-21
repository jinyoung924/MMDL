#!/usr/bin/env python
"""Print the versions that go into the report and fail early if the GPU cannot do bf16."""
import platform

import torch

print(f"python        : {platform.python_version()}")
print(f"torch         : {torch.__version__} (cuda {torch.version.cuda})")
for name in ("vllm", "transformers", "datasets", "huggingface_hub"):
    try:
        mod = __import__(name)
        print(f"{name:14}: {mod.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"{name:14}: NOT IMPORTABLE ({e})")
assert torch.cuda.is_available(), "CUDA not available"
p = torch.cuda.get_device_properties(0)
print(f"gpu           : {p.name}, {p.total_memory / 2**30:.1f} GiB, sm_{p.major}{p.minor}")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
assert p.major >= 8, "bf16 needs compute capability >= 8.0 (Ampere/Ada/Hopper) for vLLM"
