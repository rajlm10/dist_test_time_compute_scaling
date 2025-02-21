#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import re
import numpy as np
import torch

from transformers import GenerationConfig
from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores
from sal.search.utils import build_conv

# def build_conv(user_prompt, partial_answer, system_prompt):
#     # Create a conversation list
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt},
#     ]
#     if partial_answer:
#         # If we already have partial steps, treat them like assistant content
#         messages.append({"role": "assistant", "content": partial_answer})

#     # Now turn these messages into a single string
#     text_blocks = []
#     for msg in messages:
#         if msg["role"] == "system":
#             text_blocks.append(f"[SYSTEM] {msg['content']}")
#         elif msg["role"] == "user":
#             text_blocks.append(f"[USER] {msg['content']}")
#         elif msg["role"] == "assistant":
#             text_blocks.append(f"[ASSISTANT] {msg['content']}")

#     # Join them with newlines or however you want
#     full_prompt = "\n".join(text_blocks) + "\n"
#     return full_prompt


def format_with_steps(steps):
    """
    Utility to reconstruct a multi-step chain-of-thought
    from a list of step strings.
    """
    formatted = ""
    for i, step_text in enumerate(steps):
        formatted += f"## Step {i+1}: {step_text}\n\n"
    return formatted.strip()


def _hf_generate_responses(prompts, model, tokenizer, config):
    """
    Helper to generate text for each prompt using Hugging Face `model.generate`.
    Returns a list of generated strings (same length as `prompts`).
    """
    results = []

    # Define generation hyperparameters:
    generation_conf = GenerationConfig(
        max_new_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=True,        # matches vLLM sampling style
        num_return_sequences=1  # we replicate prompts for "best-of-n"
    )

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, generation_config=generation_conf)
        # Decode the entire conversation, including the prompt
        generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # If desired, you can subtract the prompt from the front to isolate the "new" text.
        # For your chain-of-thought extraction, it's okay to keep the entire text
        # because you search for "## Step" markers via regex anyway.
        results.append(generated_text)

    return results


def best_of_n(x, config: Config, model, tokenizer, prm: PRM):
    """
    Serial best-of-n approach using Hugging Face Transformers for generation,
    with multi-step chain-of-thought logic. Now uses build_conv() both
    for the initial prompt and subsequent steps to avoid mismatches.
    """

    # We assume x["problem"] is a list of user prompts; often just length=1
    problems = x["problem"]
    M = len(problems)  # number of prompts in this batch

    # 1) Build the initial conversation text for each prompt using build_conv(...).
    #    At the very start, we have no partial steps. We'll pass an empty string for `pred`.
    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in x["problem"]
    ]
    tokenizer.chat_template = config.custom_chat_template
    templated_convs = tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True
    )
    # 2) Duplicate each prompt config.n times so we can gather "n" completions per prompt
    expanded_prompts = []
    for conv_str in templated_convs:
        expanded_prompts.extend([conv_str] * config.n)
    
    start_time = time.time()

    # 3) Generate initial completions
    responses = _hf_generate_responses(expanded_prompts, model, tokenizer, config)
    if len(responses) != M * config.n:
        raise ValueError(
            f"Generated {len(responses)} responses instead of {M * config.n}"
        )
    # 4) Extract how many steps each group has, and gather the first-step completions
    pattern_steps = r"## Step \d+:\s*(.*?)(?=\n## Step \d+:|$)"
    # For each original prompt i, we have a chunk of size config.n in `responses`.
    steps_count = []
    completions_by_prompt = [[] for _ in range(M)]


    for i in range(M):
        group = responses[i * config.n : (i + 1) * config.n]
        step_lens = []
        step1_completions = []
        for resp in group:
            found_steps = re.findall(pattern_steps, resp, flags=re.DOTALL)
            step_lens.append(len(found_steps))
            if found_steps:
                step1_completions.append(found_steps[0].strip())
            else:
                step1_completions.append("")
        steps_count.append(min(step_lens))
        completions_by_prompt[i] = step1_completions
    # Score the first-step completions
    # PRM expects a list of prompts plus a list-of-lists of completions
    scores_first_step = prm.score(problems, completions_by_prompt)
    # E.g. shape might be (M, config.n). We aggregate each row:
    agg_scores = [
        [aggregate_scores(s, config.agg_strategy) for s in row]
        for row in scores_first_step
    ]
    best_indices = [np.argmax(row) for row in agg_scores]

    # For simplicity, let's assume only 1 prompt in x["problem"]:
    best_idx = best_indices[0]  # best candidate index
    # So partial best answer is step #1 from that best candidate
    pred = f"## Step 1: {completions_by_prompt[0][best_idx]}\n\n"

    max_steps = steps_count[0]

    # 5) Iterative steps from step=2 up to max_steps
    for step_id in range(1, max_steps):
        # Build conversation with the partial best answer so far
        convs = [
            build_conv(prompt,pred,config.system_prompt)
            for prompt in x["problem"]
        ]
        convs = tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=True
        )
        # We'll replicate that conv_str config.n times for "best-of-n"
        repeated_prompts = [convs] * config.n

        # Generate
        responses_step = _hf_generate_responses(repeated_prompts, model, tokenizer, config)
        if len(responses_step) != config.n:
            raise ValueError(
                f"Generated {len(responses_step)} responses instead of {config.n}"
            )

        # Parse only up to step_id + 1
        new_completions = []
        for cand_resp in responses_step:
            found_steps = re.findall(pattern_steps, cand_resp, flags=re.DOTALL)
            truncated = found_steps[: step_id + 1]
            new_completions.append(format_with_steps(truncated))

        # Score them
        # Note: we still have just one prompt in x["problem"], so problems=[problems[0]]
        step_scores = prm.score([problems[0]], [new_completions])  # shape (1, config.n)
        agg_step_scores = [aggregate_scores(s, config.agg_strategy) for s in step_scores[0]]
        best_idx_step = np.argmax(agg_step_scores)

        # Update our partial best
        pred = new_completions[best_idx_step]

    end_time = time.time() - start_time
    print(pred)
    print(f"Generation time : {end_time:.3f} seconds")

    # Using exit() like in your original code ends the entire script after each map call:
    exit()

    return pred
