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
maxExp = 6
base = 4

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
            for k in range(0,rank):
                dist.barrier()

            t0 = time.perf_counter_ns()
            dist.recv(tensor=buffer, src=i)
            logging.info("garbage %i", buffer[-1]-buffer[-2])
            start  = time.perf_counter_ns() - t0
            logging.info("recv %i -> %i # %i at %i, %f %f", i, rank, j, start, buffer[0], buffer[1])

            for k in range(rank+1,world_size):
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
                t0 = time.perf_counter_ns()
                dist.send(tensor=temp,dst=i)
                start  = time.perf_counter_ns() - t0
                logging.info("sent %i -> %i # %i at %i, %f %f", rank, i, j, start, temp[0], temp[1])

    for i in range(rank+1,world_size):
        #recv data of all ranks after
        logging.info("Accepting from rank %i", i)
        for j in range(maxExp):
            del buffer
            cuda.empty_cache()
            buffer = torch.zeros(250*(base**j)).to(device)
            for k in range(0,rank+1):
                dist.barrier()

            t0 = time.perf_counter_ns()
            dist.recv(tensor=buffer, src=i)
            logging.info("garbage %i", buffer[-1]-buffer[-2])
            start = time.perf_counter_ns() - t0
            logging.info("recv %i -> %i # %i at %i, %f %f", i, rank, j, start, buffer[0], buffer[1])

            for k in range(rank+2,world_size):
                dist.barrier()

    # dist.barrier()
    # t0 = time.time()
    # for i in range(5):
    #     size = 1000**(i+1)
    #     temp = torch.rand(size).to(device)
    #     if rank == 0:
    #         dist.barrier()
    #         logging.info("Rank 0 Start send")
    #         temp = torch.rand(size).to(device)
    #         dist.send(tensor=temp,dst=1)
    #         now = time.time() - t0
    #         logging.info("Sent 0 -> 1, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         temp = torch.rand(size).to(device)
    #         dist.send(tensor=temp,dst=2)
    #         now = time.time() - t0
    #         logging.info("Sent 0 -> 2, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         temp = torch.rand(size).to(device)
    #         dist.send(tensor=temp,dst=3)
    #         now = time.time() - t0
    #         logging.info("Sent 0 -> 3, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         dist.barrier()
    #         dist.recv(tensor=temp,src=1)
    #         now = time.time() - t0
    #         logging.info("Recv 1 -> 0, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         dist.barrier()
    #         dist.recv(tensor=temp,src=2)
    #         now = time.time() - t0
    #         logging.info("Recv 2 -> 0, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         dist.barrier()
    #         dist.recv(tensor=temp,src=3)
    #         now = time.time() - t0
    #         logging.info("Recv 3 -> 0, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #     elif rank == 1:
    #         dist.barrier()
    #         dist.recv(tensor=temp,src=0)
    #         now = time.time() - t0
    #         logging.info("Recv 0 -> 1, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         dist.barrier()
    #         logging.info("Rank 1 Start send")
    #         temp = torch.rand(size).to(device)
    #         dist.send(tensor=temp,dst=0)
    #         now = time.time() - t0
    #         logging.info("Sent 1 -> 0, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         temp = torch.rand(size).to(device)
    #         dist.send(tensor=temp,dst=2)
    #         now = time.time() - t0
    #         logging.info("Sent 1 -> 2, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         temp = torch.rand(size).to(device)
    #         dist.send(tensor=temp,dst=3)
    #         now = time.time() - t0
    #         logging.info("Sent 1 -> 3, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         dist.barrier()
    #         dist.recv(tensor=temp,src=2)
    #         now = time.time() - t0
    #         logging.info("Recv 2 -> 1, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         dist.barrier()
    #         dist.recv(tensor=temp,src=3)
    #         now = time.time() - t0
    #         logging.info("Recv 3 -> 1, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #     elif rank == 2:
    #         dist.barrier()
    #         dist.recv(tensor=temp,src=0)
    #         now = time.time() - t0
    #         logging.info("Recv 0 -> 2, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         dist.barrier()
    #         dist.recv(tensor=temp,src=1)
    #         now = time.time() - t0
    #         logging.info("Recv 1 -> 2, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         dist.barrier()
    #         logging.info("Rank 2 Start send")
    #         temp = torch.rand(size).to(device)
    #         dist.send(tensor=temp,dst=0)
    #         now = time.time() - t0
    #         logging.info("Sent 2 -> 0, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         temp = torch.rand(size).to(device)
    #         dist.send(tensor=temp,dst=1)
    #         now = time.time() - t0
    #         logging.info("Sent 2 -> 1, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         temp = torch.rand(size).to(device)
    #         dist.send(tensor=temp,dst=3)
    #         now = time.time() - t0
    #         logging.info("Sent 2 -> 3, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         dist.barrier()
    #         dist.recv(tensor=temp,src=3)
    #         now = time.time() - t0
    #         logging.info("Recv 3 -> 2, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #     elif rank == 3:
    #         dist.barrier()
    #         dist.recv(tensor=temp,src=0)
    #         now = time.time() - t0
    #         logging.info("Recv 0 -> 3, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         dist.barrier()
    #         dist.recv(tensor=temp,src=1)
    #         now = time.time() - t0
    #         logging.info("Recv 1 -> 3, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         dist.barrier()
    #         dist.recv(tensor=temp,src=2)
    #         logging.info("Recv 2 -> 3, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         dist.barrier()
    #         logging.info("Rank 3 Start send")
    #         temp = torch.rand(size).to(device)
    #         dist.send(tensor=temp,dst=0)
    #         now = time.time() - t0
    #         logging.info("Sent 3 -> 0, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         temp = torch.rand(size).to(device)
    #         dist.send(tensor=temp,dst=1)
    #         now = time.time() - t0
    #         logging.info("Sent 3 -> 1, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    #         temp = torch.rand(size).to(device)
    #         dist.send(tensor=temp,dst=2)
    #         now = time.time() - t0
    #         logging.info("Sent 3 -> 2, %f %f %f at %f", temp[0],temp[1],temp[2], now)
    
    logging.info("Rank %i done", rank)    
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
