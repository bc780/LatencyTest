import torch
import torch.distributed as dist
import torch.cuda as cuda
import logging
import os
from utilsNersc.logging import config_logging
from utilsNersc.distributed import init_workers

tensorSize = 2000000

def run(world_size, rank):
    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    node = rank//4

    time = interReduce(tensorSize,device)
    logging.info("Rank %i All Reduce: %f", rank, time)

    time  = intraReduce(tensorSize, device, node)
    logging.info("Rank %i Node Reduce %i: %f", rank, node, time)



    exit()

    

#all nodes test
def interReduce(tensorSize, device):
    before = cuda.Event(enable_timing=True)
    after = cuda.Event(enable_timing=True)

    #generate tensor
    tensor = torch.rand(tensorSize).to(device)

    #time all reduce
    dist.barrier()
    before.record()
    dist.all_reduce(tensor)
    after.record()

    return before.elapsed_time(after)



#one node test
def intraReduce(tensorSize, device, node):
    before = cuda.Event(enable_timing=True)
    after = cuda.Event(enable_timing=True)

    group = dist.new_group([node*4, node*4+1,node*4+2,node*4+3])

    #generate tensor
    tensor = torch.rand(tensorSize).to(device)

    #time all reduce
    dist.barrier()
    before.record()
    dist.all_reduce(tensor, group = group)
    after.record()

    return before.elapsed_time(after)
    


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

