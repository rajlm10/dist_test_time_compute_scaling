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

from transformers import AutoModelForCausalLM, AutoTokenizer

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search.serial_best_of_n import best_of_n  # updated serial version
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

def main():
    # set_seeds(12)
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    logger.info("Running serial generation on one GPU.")

    # 1. Load model & tokenizer.
    logger.info(f"Loading HF model from {config.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # 2. Load reward model.
    prm = load_prm(config)

    # 3. Get dataset (for example, just the first sample).
    dataset = get_dataset(config)
    dataset = dataset.select([0])

    # 4. Process each sample serially.
    results = []
    for sample in dataset:
        result = best_of_n(sample, config=config, model=model, tokenizer=tokenizer, prm=prm)
        results.append(result)

    # Optionally, score the results.
    # results = score(results, config)

    # Optionally, log and/or save final results.
    # logger.info("Final results: {}".format(results))
    # save_dataset(results, config)

    logger.info("Done 🔥!")
    return results

if __name__ == "__main__":
    main()
