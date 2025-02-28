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

import logging
import torch
import os
import torch.distributed as dist

import torch.distributed
from transformers import AutoModelForCausalLM, AutoTokenizer

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search.fault_tolerant_bofn import best_of_n
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def set_seeds(seed):
    import numpy as np
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU random seed
    torch.cuda.manual_seed(seed)  # GPU random seed
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ddp_setup():
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def main():
    set_seeds(12)
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    # Initialize distributed process group if using multiple GPUs.
    if config.n > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        # Set each process to use a unique GPU (env variable LOCAL_RANK is set by torchrun)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    
    # Print rank and PID for each process.
    rank = dist.get_rank() if dist.is_initialized() else 0
    pid = os.getpid()
    logger.info(f"Process rank {rank} running on GPU {torch.cuda.current_device()} with PID: {pid}")

    logger.info(f"Distributed initialized: rank {rank} on GPU {dist.get_rank()} of {dist.get_world_size()}")

    approach_fn = best_of_n

    # 1. Load model & tokenizer (each process loads its own copy)
    logger.info(f"Loading HF model from {config.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # 2. Load reward model
    prm = load_prm(config)

    # 3. Get dataset (for example, just the first sample)
    dataset = get_dataset(config)
    dataset = dataset.select([0])

    # 4. Iterate over the dataset without using map.
    results = []
    for sample in dataset:
        result = approach_fn(sample, config=config, model=model, tokenizer=tokenizer, prm=prm)
        if rank==0:
            print(result)
            if os.path.exists("checkpoint.json"):
                os.remove("checkpoint.json")

        results.append(result)
    
    # 5. Optionally score the results.
    # results = score(results, config)

    # 6. Save or log final dataset (only rank 0 does logging)
    # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    #     logger.info("Final results: {}".format(results))
    # Optionally, save the dataset:
    # save_dataset(results, config)

    logger.info("Done 🔥!")
    return results

if __name__ == "__main__":
    # Uncomment if you want to explicitly set up DDP.
    # ddp_setup()
    main()
    torch.distributed.destroy_process_group()
