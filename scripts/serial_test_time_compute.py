#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
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

from transformers import AutoModelForCausalLM, AutoTokenizer

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search.serial_best_of_n import best_of_n
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

APPROACHES = {
    "best_of_n": best_of_n,
}

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

def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    approach_fn = APPROACHES[config.approach]

    # 1. Load model & tokenizer
    logger.info(f"Loading HF model from {config.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()  # Move model to GPU if available

    # 2. Load reward model
    prm = load_prm(config)

    # 3. Get dataset (example: just the first sample)
    dataset = get_dataset(config)
    dataset = dataset.select([0])

    # 4. Map across the dataset with your approach function
    #    (serial, single-GPU approach)
    dataset = dataset.map(
        approach_fn,
        batched=True,
        batch_size=config.search_batch_size,
        fn_kwargs={"config": config, "model": model, "tokenizer": tokenizer, "prm": prm},
        desc="Running search",
        load_from_cache_file=False,
    )

    # 5. Optionally score the dataset
    dataset = score(dataset, config)

    # 6. Save or log final dataset (commented out by default)
    # save_dataset(dataset, config)

    logger.info("Done 🔥!")


if __name__ == "__main__":
    set_seeds(12)
    main()