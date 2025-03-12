import logging
import os
import threading
import time

import torch
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

WORLD_SIZE=5 # total gpus + 1

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

def ddp_setup(rank, world_size):
    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    # torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(rank)

class Worker:
    def heartbeat(self, interval=5):
        while True:
            heartbeat_ts = torch.tensor([
                self.rank, 
                time.time(),
                torch.cuda.utilization(), 
                len(self.tasks)
                ], dtype=torch.int64, device=self.device
            ) 
            dist.send(heartbeat_ts, dst=0) # aka send to master
            
            time.sleep(interval)
            
    def __init__(self, rank, world_size):
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.tasks = []
        ddp_setup(rank, world_size)
        threading.Thread(target=heartbeat, args=(rank,), daemon=True).start()
        
    def process_task(self, task):
        # process some output and send back to master
        return 0
    
    


class Master:
    def __init__(self, world_size):
        self.rank = 0
        self.device = torch.device("cuda:0")
        self.tasks=[]
        ddp_setup(self.rank, world_size)
        self.heartbeats = torch.zeros((world_size, 4), dtype=torch.int64,device="cuda:0")
        
    def master(self, world_size):
        
        while True:
            # confirm functionality first
            # think of load balancing methodology later
            for worker in range(1, world_size):
                if self.tasks:
                    task = self.tasks.pop()
                    dist.send(task, dst=worker)
                    
            for worker in range(1, world_size):
                try:
                    heartbeat = torch.zeros(4, dtype=torch.int64, device="cuda:0")
                    dist.recv(heartbeat, src=worker, timeout=10)
                    self.heartbeats[worker] = heartbeat
                    
                    # could add more logic here to compare timestamp in the heartbeat object vs the local timestamp
                except RuntimeError:
                    print(f"Node {worker} failed")