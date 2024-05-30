import torch
import torch.distributed as dist
# import argparse
import logging
import os
import time

from utilsNersc.logging import config_logging
from utilsNersc.distributed import init_workers, try_barrier


# max and base of exponents to modify data size
maxExp = 8
base = 4

def run(world_size, rank):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = torch.zeros(250*(base**maxExp)).to(device)
    for i in range(0,rank):
        #recv data of all ranks before
        logging.info("Accepting from rank %i", i)
        for j in range(maxExp):
            logging.info("Hit barrier")
            try_barrier()
            dist.recv(tensor=tensor,src=i, tag=i*world_size+j)
            start  = time.time()
            logging.info("recv %i -> %i # %i at %f", i, rank, j, start)
    for j in range(maxExp):
        logging.info("Hit barrier")
        try_barrier()
        for i in range(world_size):
            #send data to all ranks
            #ensure it does not send to itself
            if i != rank:           
            #generate random tensor of correct size
                temp = torch.rand(250*(base**j)).to(device)
                dist.send(tensor=temp,dst=i,tag=rank*world_size+j)
                start  = time.time()
                logging.info("sent %i -> %i # %i at %f", rank, i, j, start)
    for i in range(rank+1,world_size):
        #recv data of all ranks after
        logging.info("Accepting from rank %i", i)
        for j in range(maxExp):
            try_barrier()
            dist.recv(tensor=tensor,src=i,tag=i*world_size+j)
            start = time.time()
            logging.info("recv %i -> %i # %i at %f", i, rank, j, start)

    exit()

def main():
    # parser = argparse.ArgumentParser()

    rank, n_ranks = init_workers("nccl")

    output_dir = "$SCRATCH/bandwidthTest/output"
    output_dir = os.path.expandvars(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_file = (os.path.join(output_dir, 'out_%i.log' % rank)
                if output_dir is not None else None)
    config_logging(verbose=True, log_file=log_file)
    logging.info('Initialized rank %i out of %i', rank, n_ranks)

    run(n_ranks, rank)
    exit()

    

if __name__ == "__main__":
    main()
