import torch
import torch.distributed as dist
import torch.cuda as cuda
import logging
import os
from utilsNersc.logging import config_logging
from utilsNersc.distributed import init_workers

tensorSize = 8000000

def run(world_size, rank):

    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    test = torch.zeros(2000000000).to(device)
    logging.info("Tensor size: %i",test.element_size()*test.nelement())
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

