#!/usr/bin/env python

import logging
import torch
import os
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoTokenizer
from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search.batch_best_of_n import best_of_n
from sal.utils.data import get_dataset
from sal.utils.parser import H4ArgumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

APPROACHES = {"batch_best_of_n": best_of_n}

def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    if config.n > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    approach_fn = APPROACHES[config.approach]

    logger.info(f"Loading model from {config.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    prm = load_prm(config)

    # Load dataset and select a batch of queries
    dataset = get_dataset(config)
    batch_size = config.batch_size  # Assuming `batch_size` is defined in config
    
    dataset = dataset.select(range(batch_size))  # Select the first batch_size queries

    results = []
    for i in range(0, len(dataset), batch_size):
        # Python handles out-of-range slicing gracefully
        batch = dataset[i : i + batch_size]
        batch_results = approach_fn(batch, config=config, model=model, tokenizer=tokenizer, prm=prm)
        results.extend(batch_results)

    logger.info("Done processing batch 🔥!")
    return results

if __name__ == "__main__":
    main()
    torch.distributed.destroy_process_group()