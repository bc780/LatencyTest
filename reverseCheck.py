import torch
import torch.distributed as dist
import torch.cuda as cuda
# import argparse
import logging
import os
import time

from utilsNersc.logging import config_logging
from utilsNersc.distributed import init_workers

maxExp = 1
base = 8000000

def run(world_size, rank):
    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    dist.barrier()
    buffer = torch.zeros(1000)

    for i in range(world_size-1,rank, -1):
        #recv data of all ranks before
        logging.info("Accepting from rank %i", i)
        for j in range(maxExp):
            del buffer
            cuda.empty_cache()
            buffer = torch.zeros(250*(base**(j+1))).to(device)
            for k in range(world_size-2,rank,-1):
                dist.barrier()
                dist.barrier()

            dist.barrier()
            before = cuda.Event(enable_timing=True)
            after = cuda.Event(enable_timing=True)
            
            # before = time.perf_counter_ns()
            before.record()
            dist.recv(tensor=buffer, src=i)
            # after = time.perf_counter_ns()
            after.record()
            checksum = buffer[-1] + buffer[-2] + buffer[-3]
            logging.info("CHECKSUM %i %i %i: %f", i, rank, j, checksum)
            dist.barrier()

            # start  = after - before 
            start = before.elapsed_time(after)
            start = int(start*1000000)
            logging.info("recv reverse %i -> %i # %i at %i, %f %f", i, rank, j, start, buffer[0], buffer[1])

            for k in range(rank-1,-1,-1):
                dist.barrier()
                dist.barrier()
            
    
    logging.info("Sending from rank %i", rank)
    for j in range(maxExp):
        
        for i in range(world_size-1, -1, -1):
            #send data to all ranks
            #ensure it does not send to itself
            if i != rank:           
            #generate random tensor of correct size
                temp = torch.rand(250*(base**(j+1))).to(device)
                dist.barrier()

                before = cuda.Event(enable_timing=True)
                after = cuda.Event(enable_timing=True)

                # before = time.perf_counter_ns()
                before.record()
                dist.send(tensor=temp,dst=i)
                # after = time.perf_counter_ns()
                after.record()
                checksum = temp[-1] + temp[-2] + temp[-3]
                logging.info("CHECKSUM %i %i %i: %f", rank, i, j, checksum)
                dist.barrier()

                # start  = after-before
                start = before.elapsed_time(after)
                start = int(start*1000000)
                logging.info("sent reverse %i -> %i # %i at %i, %f %f", rank, i, j, start, temp[0], temp[1])
    del temp
    cuda.empty_cache()

    for i in range(rank-1,-1,-1):
        #recv data of all ranks after
        logging.info("Accepting from rank %i", i)
        for j in range(maxExp):
            del buffer
            cuda.empty_cache()
            buffer = torch.zeros(250*(base**(j+1))).to(device)
            for k in range(world_size-1,rank,-1):
                dist.barrier()
                dist.barrier()

            dist.barrier()
            before = cuda.Event(enable_timing=True)
            after = cuda.Event(enable_timing=True)

            # before = time.perf_counter_ns()
            before.record()
            dist.recv(tensor=buffer, src=i)
            # after = time.perf_counter_ns()
            after.record()
            checksum = buffer[-1] + buffer[-2] + buffer[-3]
            logging.info("CHECKSUM %i %i %i: %f", i, rank, j, checksum)
            dist.barrier()

            # start = after - before
            start = before.elapsed_time(after)
            start = int(start*1000000)
            logging.info("recv reverse %i -> %i # %i at %i, %f %f", i, rank, j, start, buffer[0], buffer[1])

            for k in range(rank-2,-1,-1):
                dist.barrier()
                dist.barrier()

    


def main():

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
