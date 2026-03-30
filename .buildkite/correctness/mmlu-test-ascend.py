# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import json
import os
import time
from pathlib import Path

# Third Party
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, set_seed
from tqdm import tqdm

# vLLM
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# setting PYTHONHASHSEED derandomizes token chunking
os.environ["PYTHONHASHSEED"] = "0"

choices = ["A", "B", "C", "D"]
TOKEN_TUNING_UNITS = [
    " padding",
    " ignore",
    " cache",
    " reuse",
    ".",
    "\n",
]

tokenizer = None


def count_tokens(text: str) -> int:
    assert tokenizer is not None
    return len(tokenizer.encode(text, add_special_tokens=False))


def fit_text_to_target_tokens(
    prefix: str,
    suffix: str,
    filler_unit: str,
    target_tokens: int,
) -> str:
    """Build text whose token count matches target_tokens exactly."""
    base_tokens = count_tokens(prefix + suffix)
    if base_tokens > target_tokens:
        raise ValueError(
            f"Base prompt already uses {base_tokens} tokens, exceeds target "
            f"{target_tokens}."
        )

    low = 0
    high = 1
    while count_tokens(prefix + filler_unit * high + suffix) <= target_tokens:
        high *= 2

    best_repeat = 0
    while low <= high:
        mid = (low + high) // 2
        current_tokens = count_tokens(prefix + filler_unit * mid + suffix)
        if current_tokens <= target_tokens:
            best_repeat = mid
            low = mid + 1
        else:
            high = mid - 1

    filler = filler_unit * best_repeat
    text = prefix + filler + suffix
    current_tokens = count_tokens(text)
    if current_tokens == target_tokens:
        return text

    while current_tokens < target_tokens:
        appended = False
        for unit in TOKEN_TUNING_UNITS:
            candidate_filler = filler + unit
            candidate = prefix + candidate_filler + suffix
            candidate_tokens = count_tokens(candidate)
            if candidate_tokens <= target_tokens:
                filler = candidate_filler
                text = candidate
                current_tokens = candidate_tokens
                appended = True
                if current_tokens == target_tokens:
                    return text
        if not appended:
            break

    if current_tokens != target_tokens:
        raise ValueError(
            f"Unable to tune prompt to exactly {target_tokens} tokens; got "
            f"{current_tokens}."
        )
    return text


def build_front_loaded_prompt(
    subject_raw: str,
    question_idx: int,
    test_df: pd.DataFrame,
    target_prompt_tokens: int,
) -> str:
    question = test_df.iloc[question_idx, 0]
    prompt_id = f"{subject_raw}-{question_idx:04d}"

    prefix = (
        "You are solving a multiple choice question.\n"
        "Return exactly one capital letter: A, B, C, or D.\n"
        "Do not include reasoning or any extra text.\n"
        f"Question ID: {prompt_id}\n"
        f"Subject: {subject_raw.replace('_', ' ')}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Options:\n"
    )

    num_options = test_df.shape[1] - 2
    for option_idx in range(num_options):
        prefix += (
            f"{choices[option_idx]}. {test_df.iloc[question_idx, option_idx + 1]}\n"
        )

    prefix += (
        "\nThe answer must be derived from the question and options above.\n"
        "The following padding block is only for cache reuse testing and must be "
        "ignored.\n\n"
        "Padding Block Start\n"
    )

    suffix = "\nPadding Block End\nAnswer:"

    filler_unit = (
        f" Ignore this cache-reuse padding for sample {prompt_id}. "
        "It does not add any information needed to answer the question. "
    )

    return fit_text_to_target_tokens(
        prefix=prefix,
        suffix=suffix,
        filler_unit=filler_unit,
        target_tokens=target_prompt_tokens,
    )


def extract_prediction(output_text: str) -> str:
    stripped = output_text.strip()
    if stripped and stripped[0] in choices:
        return stripped[0]
    for char in stripped:
        if char in choices:
            return char
    return "A"


def run_round(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    sleep_seconds: float,
) -> list[str]:
    outputs = []
    for prompt in prompts:
        result = llm.generate(prompt, sampling_params, use_tqdm=False)
        outputs.append(result[0].outputs[0].text)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return outputs


def build_llm(args) -> LLM:
    kv_transfer_config = None
    if args.use_lmcache:
        kv_transfer_config = KVTransferConfig(
            kv_connector=args.kv_connector,
            kv_role=args.kv_role,
            kv_connector_module_path=args.kv_connector_module_path,
        )

    compilation_config = (
        json.loads(args.compilation_config_json)
        if args.compilation_config_json
        else None
    )
    additional_config = (
        json.loads(args.additional_config_json)
        if args.additional_config_json
        else None
    )

    return LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_transfer_config=kv_transfer_config,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        seed=args.seed,
        dtype=args.dtype,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        async_scheduling=args.async_scheduling,
        disable_hybrid_kv_cache_manager=args.disable_hybrid_kv_cache_manager,
        compilation_config=compilation_config,
        additional_config=additional_config,
        prefix_caching_hash_algo=args.prefix_caching_hash_algo,
    )


def main(args):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = build_llm(args)

    script_dir = Path(__file__).resolve().parent
    data_root = script_dir / "data"
    dev_root = data_root / "dev"
    test_root = data_root / "test"

    test_files = sorted(test_root.glob("*_test.csv"))
    subjects = [path.name.split("_test.csv")[0] for path in test_files]

    prompts = []
    prompt_subjects = []
    labels = []
    prompt_lengths = []

    subject_prompt_lengths = {}
    subject_labels = {}

    for subject_raw in tqdm(
        subjects[: args.number_of_subjects],
        desc="Building prompts",
    ):
        _ = dev_root / f"{subject_raw}_dev.csv"
        test_df = pd.read_csv(test_root / f"{subject_raw}_test.csv", header=None)
        subject_labels[subject_raw] = []
        subject_prompt_lengths[subject_raw] = []

        for i in range(test_df.shape[0]):
            prompt = build_front_loaded_prompt(
                subject_raw=subject_raw,
                question_idx=i,
                test_df=test_df,
                target_prompt_tokens=args.target_prompt_tokens,
            )
            label = test_df.iloc[i, test_df.shape[1] - 1]
            prompt_len = count_tokens(prompt)

            prompts.append(prompt)
            prompt_subjects.append(subject_raw)
            labels.append(label)
            prompt_lengths.append(prompt_len)
            subject_labels[subject_raw].append(label)
            subject_prompt_lengths[subject_raw].append(prompt_len)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=2,
        seed=args.seed,
        n=1,
        stop=None,
    )

    if args.use_lmcache:
        run_round(llm, prompts, sampling_params, args.sleep_seconds)
        if args.pass_sleep_seconds > 0:
            time.sleep(args.pass_sleep_seconds)

    outputs = run_round(llm, prompts, sampling_params, args.sleep_seconds)
    predictions = [extract_prediction(text) for text in outputs]

    accuracies = []
    num_questions = []
    output_dict = {
        "config": {
            "model": args.model,
            "use_lmcache": args.use_lmcache,
            "target_prompt_tokens": args.target_prompt_tokens,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
            "enable_prefix_caching": args.enable_prefix_caching,
            "seed": args.seed,
        }
    }

    prediction_cursor = 0
    for subject_raw in subjects[: args.number_of_subjects]:
        subject_num_questions = len(subject_labels[subject_raw])
        subject_predictions = predictions[
            prediction_cursor : prediction_cursor + subject_num_questions
        ]
        prediction_cursor += subject_num_questions

        accuracy = np.mean(
            np.array(subject_predictions) == np.array(subject_labels[subject_raw])
        )
        accuracies.append(accuracy)
        num_questions.append(subject_num_questions)
        output_dict[subject_raw] = {
            "accuracy": accuracy,
            "num_questions": subject_num_questions,
            "prompt_min_tokens": min(subject_prompt_lengths[subject_raw]),
            "prompt_max_tokens": max(subject_prompt_lengths[subject_raw]),
            "prompt_avg_tokens": float(np.mean(subject_prompt_lengths[subject_raw])),
        }

    total_accuracy = np.mean(accuracies) if accuracies else 0.0
    total_num_questions = sum(num_questions)
    output_dict["total"] = {
        "accuracy": total_accuracy,
        "num_questions": total_num_questions,
        "prompt_min_tokens": min(prompt_lengths) if prompt_lengths else 0,
        "prompt_max_tokens": max(prompt_lengths) if prompt_lengths else 0,
        "prompt_avg_tokens": float(np.mean(prompt_lengths)) if prompt_lengths else 0.0,
    }

    result_path = Path(args.result_file)
    with result_path.open("w", encoding="utf-8") as file_obj:
        for subject, value in output_dict.items():
            file_obj.write(json.dumps({subject: value}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--number-of-subjects", type=int, default=25)
    parser.add_argument("--use-lmcache", action="store_true", default=False)
    parser.add_argument("--target-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prefix-caching-hash-algo",
        type=str,
        default="sha256",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--disable-hybrid-kv-cache-manager",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--pass-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--kv-connector",
        type=str,
        default="LMCacheAscendConnectorV1Dynamic",
    )
    parser.add_argument(
        "--kv-connector-module-path",
        type=str,
        default="lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1",
    )
    parser.add_argument("--kv-role", type=str, default="kv_both")
    parser.add_argument("--compilation-config-json", type=str)
    parser.add_argument("--additional-config-json", type=str)
    parser.add_argument("--result-file", type=str)
    parsed_args = parser.parse_args()

    if parsed_args.result_file is None:
        suffix = "lmcache" if parsed_args.use_lmcache else "baseline"
        parsed_args.result_file = (
            f"{parsed_args.model.split('/')[-1]}-ascend-{suffix}.jsonl"
        )

    main(parsed_args)
