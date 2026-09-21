"""vLLM engine construction. Kept separate so a fine-tuned checkpoint (local dir or hub id)
can be swapped in by changing only ``model.path``/``model.revision``."""
import os


def build_engine(model_path: str, revision: str | None, dtype: str, max_model_len: int,
                 gpu_memory_utilization: float, limit_images: int, seed: int):
    from vllm import LLM

    kwargs = dict(
        model=model_path,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": limit_images},
        seed=seed,
        trust_remote_code=False,
    )
    if revision and not os.path.isdir(model_path):
        kwargs["revision"] = revision
        kwargs["tokenizer_revision"] = revision
    return LLM(**kwargs)


def build_sampling_params(sampling: dict, max_new_tokens: int, seed: int):
    from vllm import SamplingParams

    if not sampling.get("do_sample", True):
        return SamplingParams(temperature=0.0, max_tokens=max_new_tokens, seed=seed,
                              repetition_penalty=sampling.get("repetition_penalty", 1.0))
    return SamplingParams(
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        top_k=sampling["top_k"],
        repetition_penalty=sampling["repetition_penalty"],
        presence_penalty=sampling["presence_penalty"],
        max_tokens=max_new_tokens,
        seed=seed,
    )
