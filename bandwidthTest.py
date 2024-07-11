import torch
import torch.distributed as dist
import torch.cuda as cuda
# import argparse
import logging
import os
import time

from utilsNersc.logging import config_logging
from utilsNersc.distributed import init_workers


# max and base of exponents to modify data size
maxExp = 1
base = 65536

def run(world_size, rank):
    

    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    dist.barrier()
    logging.info("Sync time")
    buffer = torch.zeros(1000)
    for i in range(0,rank):
        #recv data of all ranks before
        logging.info("Accepting from rank %i", i)
        for j in range(maxExp):
            del buffer
            cuda.empty_cache()
            buffer = torch.zeros(250*(base**j)).to(device)
            for k in range(0,rank-1):
                dist.barrier()
                dist.barrier()

            dist.barrier()
            # before = cuda.Event(enable_timing=True)
            # after = cuda.Event(enable_timing=True)
            
            before = time.perf_counter_ns()
            dist.recv(tensor=buffer, src=i)
            checksum = buffer[-1] + buffer[-2] + buffer[-3]
            logging.info("CHECKSUM %i %i %i: %f", i, rank, j, checksum)
            dist.barrier()
            after = time.perf_counter_ns()

            start  = after - before
            logging.info("recv %i -> %i # %i at %i, %f %f", i, rank, j, start, buffer[0], buffer[1])

            for k in range(rank+1,world_size):
                dist.barrier()
                dist.barrier()
            
    
    logging.info("Sending from rank %i", rank)
    for j in range(maxExp):
        temp = torch.rand(250*(base**j)).to(device)
        
        for i in range(world_size):
            #send data to all ranks
            #ensure it does not send to itself
            if i != rank:           
            #generate random tensor of correct size
                dist.barrier()

                # before = cuda.Event(enable_timing=True)
                # after = cuda.Event(enable_timing=True)

                before = time.perf_counter_ns()
                dist.send(tensor=temp,dst=i)
                after = time.perf_counter_ns()
                checksum = temp[-1] + temp[-2] + temp[-3]
                logging.info("CHECKSUM %i %i %i: %f", rank, i, j, checksum)
                dist.barrier()

                start  = after-before
                logging.info("sent %i -> %i # %i at %i, %f %f", rank, i, j, start, temp[0], temp[1])

    for i in range(rank+1,world_size):
        #recv data of all ranks after
        logging.info("Accepting from rank %i", i)
        for j in range(maxExp):
            del buffer
            cuda.empty_cache()
            buffer = torch.zeros(250*(base**j)).to(device)
            for k in range(0,rank):
                dist.barrier()
                dist.barrier()


            dist.barrier()
            # before = cuda.Event(enable_timing=True)
            # after = cuda.Event(enable_timing=True)

            before = time.perf_counter_ns()
            dist.recv(tensor=buffer, src=i)
            checksum = buffer[-1] + buffer[-2] + buffer[-3]
            logging.info("CHECKSUM %i %i %i: %f", i, rank, j, checksum)
            dist.barrier()
            after = time.perf_counter_ns()

            start = after - before
            logging.info("recv %i -> %i # %i at %i, %f %f", i, rank, j, start, buffer[0], buffer[1])

            for k in range(rank+2,world_size):
                dist.barrier()
                dist.barrier()
    
    logging.info("Rank %i start All-Reduce", rank)
    node = rank//4
    for i in range(0,node):
        dist.barrier()
        temp = dist.new_group([i*4, i*4+1,i*4+2,i*4+3])
        dist.barrier()
    
    tensor = torch.rand(250*(base**j)).to(device)
    logging.info("Rank %i HIT BARRIER", rank)
    dist.barrier()
    before = time.perf_counter_ns()
    group = dist.new_group([node*4, node*4+1,node*4+2,node*4+3])
    logging.info("Made group")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    checksum = tensor[-1] + tensor[-2] + tensor[-3]
    logging.info("Node Reduce CHECKSUM %i: %f", rank, checksum)
    dist.barrier()
    after = time.perf_counter_ns

    start = after - before
    logging.info("Node Reduce rank %i at %i", rank, start)

    for i in range(node+1, world_size/4):
        dist.barrier()
        temp = dist.new_group([i*4, i*4+1,i*4+2,i*4+3])
        dist.barrier()
    
    tensor = torch.rand(250*(base**j)).to(device)
    dist.barrier()
    before = time.perf_counter_ns()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    checksum = tensor[-1] + tensor[-2] + tensor[-3]
    logging.info("All Reduce CHECKSUM %i: %f", checksum)
    dist.barrier()
    after = time.perf_counter_ns

    start = after-before
    logging.info("All Reduce rank %i at %i", rank, start)
    
    logging.info("Rank %i done", rank)    
    exit()

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
