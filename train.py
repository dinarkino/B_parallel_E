import time

import hydra
from mpi4py import MPI
from omegaconf import DictConfig

import bpe


@hydra.main(config_name="config")
def main(cfg: DictConfig):
    comm = MPI.COMM_WORLD

    bpe_obj = bpe.BPE(vocab_size=cfg.vocab_size, max_iters=cfg.max_iters, bpe_path=cfg.bpe_path, comm=comm)

    with open(hydra.utils.to_absolute_path(cfg.text_path), "r") as f:
        text = f.read()

    if comm.Get_rank() == 0:
        start_time = time.time()
    bpe_obj.train(text)
    if comm.Get_rank() == 0:
        print()
        print(time.time() - start_time)


if __name__ == "__main__":
    main()
