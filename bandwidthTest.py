import torch
import torch.distributed as dist
import argparse
import logging
import os

from utilsNersc.logging import config_logging
from utilsNersc.distributed import init_workers

def run(world_size, rank):
    tensor = torch.zeros(1)
    for i in range(0,rank):
        #recv data of all ranks before
        for j in range(8):
            dist.recv(tensor=tensor,src=i)
    for i in range(world_size):
        #send data to all ranks
        #ensure it does not send to itself
        if i != rank:
            for j in range(8):
            #generate random tensor of correct size
                tensor = torch.rand(250*(10**j))
                dist.send(tensor=tensor,dst=i)
    for i in range(rank+1,world_size):
        #recv data of all ranks after
        for j in range(8):
            dist.recv(tensor=tensor,src=i)

def main():
    parser = argparse.ArgumentParser()

    rank, n_ranks = init_workers("nccl")

    output_dir = "$SCRATCH/bandwidthTest/output"
    output_dir = os.path.expandvars(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_file = (os.path.join(output_dir, 'out_%i.log' % rank)
                if output_dir is not None else None)
    config_logging(verbose=True, log_file=log_file)
    logging.info('Initialized rank %i out of %i', rank, n_ranks)

    run(n_ranks, rank)

    

if __name__ == "__main__":
    main()
