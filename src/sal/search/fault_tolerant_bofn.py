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
import json
import os
import numpy as np
import torch
import torch.distributed as dist

from transformers import GenerationConfig
from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

def save_checkpoint(step, best_candidate, max_steps, checkpoint_path="checkpoint.json"):
    """Save the current step, max_steps and best candidate to a checkpoint file."""
    checkpoint_data = {
        "step": step,
        "best_candidate": best_candidate,
        "max_steps": max_steps,
    }
    # Write atomically.
    temp_path = checkpoint_path + ".tmp"
    with open(temp_path, "w") as f:
        json.dump(checkpoint_data, f)
    os.replace(temp_path, checkpoint_path)

def load_checkpoint(checkpoint_path="checkpoint.json"):
    """
    Load checkpoint if available.
    Returns (step, best_candidate, max_steps) or (None, None, None).
    If "max_steps" is missing, it defaults to None.
    """
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)
        step = checkpoint_data.get("step", None)
        best_candidate = checkpoint_data.get("best_candidate", None)
        max_steps = checkpoint_data.get("max_steps", None)
        return step, best_candidate, max_steps
    return None, None, None

def build_conv(user_prompt, partial_answer, system_prompt):
    """
    Build a conversation as a list of messages. 
    Each message is a dict with 'role' and 'content'.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if partial_answer:
        messages.append({"role": "assistant", "content": partial_answer})
    return messages  # Return list of dicts, not a formatted string.

def format_with_steps(steps):
    """
    Reconstruct a multi-step chain-of-thought from a list of step strings.
    """
    formatted = ""
    for i, step_text in enumerate(steps):
        formatted += f"## Step {i+1}: {step_text}\n\n"
    return formatted.strip()

def _hf_generate_responses(prompts, model, tokenizer, config):
    """
    Generate text for each prompt using Hugging Face model.generate.
    Returns a list of generated strings (one per prompt).
    """
    results = []
    generation_conf = GenerationConfig(
        max_new_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=True,
        num_return_sequences=1,
    )
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, generation_config=generation_conf)
        generated_text = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        results.append(generated_text)
    return results

def best_of_n(x, config: Config, model, tokenizer, prm: PRM):
    """
    Distributed best-of-n approach with checkpointing.
    Each GPU (process) generates one candidate completion per step.
    Rank 0 gathers all candidate completions, ranks them using PRM,
    picks the best candidate, saves a checkpoint, and then broadcasts the partial answer
    for the next step.
    
    The maximum number of steps is determined dynamically from the candidates produced in the
    initial generation step.
    """
    # Initialize the process group if needed.
    if config.n > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # If the job is restarted with fewer GPUs, update config.n automatically.
    if world_size < config.n:
        print(f"WARNING: Resuming with {world_size} GPUs instead of originally configured {config.n}")
        config.n = world_size

    # Print rank and PID for debugging.
    pid = os.getpid()
    print(f"[Rank {rank}] Process PID: {pid}")

    # Expecting x["problem"] to be a list with a single user prompt.
    problem = x["problem"]

    # 1) Build the initial conversation prompt.
    init_conv = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": problem},
    ]
    conv = tokenizer.apply_chat_template(
        [init_conv],
        tokenize=False,
        add_generation_prompt=True
    )[0]

    start_time = time.time()
    checkpoint_path = "checkpoint.json"

    # 2) Check for an existing checkpoint.
    checkpoint_step, pred, max_steps = load_checkpoint(checkpoint_path)

    # Early exit: if all steps are complete, resume without further generation.
    if checkpoint_step is not None and checkpoint_step == max_steps:
        if rank == 0:
            print(f"All steps completed (step {checkpoint_step}/{max_steps}). Resuming without further generation.")
        return pred

    if checkpoint_step is None or max_steps is None:
        # No checkpoint or incomplete checkpoint: run the initial generation step.
        responses = _hf_generate_responses([conv], model, tokenizer, config)
        my_response = responses[0]
        candidates = [None for _ in range(world_size)]
        dist.all_gather_object(candidates, my_response)

        if rank == 0:
            pattern_steps = r"## Step \d+:\s*(.*?)(?=\n## Step \d+:|$)"
            step_lens = []
            step1_completions = []
            for resp in candidates:
                found_steps = re.findall(pattern_steps, resp, flags=re.DOTALL)
                step_lens.append(len(found_steps))
                step1_completions.append(found_steps[0].strip() if found_steps else "")
            # Dynamically determine the maximum number of steps available.
            max_steps = min(step_lens)
            scores_first_step = prm.score([problem], [step1_completions])
            agg_scores = [aggregate_scores(s, config.agg_strategy) for s in scores_first_step[0]]
            best_idx = int(np.argmax(agg_scores))
            pred = f"## Step 1: {step1_completions[best_idx]}\n\n"
            # Save checkpoint for step 1.
            save_checkpoint(1, pred, max_steps, checkpoint_path)
            current_step = 1
        else:
            current_step = None
        best_data = [pred, current_step, max_steps]
        dist.broadcast_object_list(best_data, src=0)
        pred, current_step, max_steps = best_data
    else:
        # Resume from checkpoint.
        current_step = checkpoint_step
        if rank == 0:
            print(f"Resuming from checkpoint at step {current_step} (max steps: {max_steps})")
        best_data = [pred, max_steps]
        dist.broadcast_object_list(best_data, src=0)
        pred, max_steps = best_data

    # 3) Iteratively generate subsequent steps.
    for step_id in range(current_step, max_steps):
        conv_messages = build_conv(problem, pred, config.system_prompt)
        conv = tokenizer.apply_chat_template(
            [conv_messages], tokenize=False, add_generation_prompt=True
        )[0]

        responses_step = _hf_generate_responses([conv], model, tokenizer, config)
        my_resp_step = responses_step[0]
        step_candidates = [None for _ in range(world_size)]
        dist.all_gather_object(step_candidates, my_resp_step)

        if rank == 0:
            new_completions = []
            pattern_steps = r"## Step \d+:\s*(.*?)(?=\n## Step \d+:|$)"
            for cand_resp in step_candidates:
                cand_resp = pred + cand_resp
                found_steps = re.findall(pattern_steps, cand_resp, flags=re.DOTALL)
                truncated = found_steps[:step_id + 1]
                new_completions.append(format_with_steps(truncated))
            scores = prm.score([problem], [new_completions])
            agg_scores = [aggregate_scores(s, config.agg_strategy) for s in scores[0]]
            best_idx_step = int(np.argmax(agg_scores))
            pred = new_completions[best_idx_step]
            save_checkpoint(step_id + 1, pred, max_steps, checkpoint_path)

        best_data = [pred]
        dist.broadcast_object_list(best_data, src=0)
        pred = best_data[0]
    if rank==0:
        print(f"[Rank {rank}] Generation complete. Total time: {time.time() - start_time:.3f} seconds")
    return pred
