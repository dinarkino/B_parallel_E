import time
import os

import hydra
from mpi4py import MPI
from omegaconf import DictConfig

import bpe

TEXT_SEND_TAG = 1

@hydra.main(config_name="configs/config")
def main(cfg: DictConfig):
    comm = MPI.COMM_WORLD

    bpe_obj = bpe.BPE(max_iters=cfg.max_iters,
                      vocab_size=cfg.vocab_size,
                      tokens_path=cfg.tokens_path,
                      id2token_path=cfg.id2token_path,
                      encodings_path=cfg.encodings_path,
                      comm=comm,
                      store_files=cfg.store_files)

    if comm.Get_rank() == 0:
        with open(hydra.utils.to_absolute_path(cfg.text_path), "r") as f:
            text = f.read()
        step = len(text) // comm.Get_size() + 1
        for i in range(1, comm.Get_size()):
            comm.send(text[step * i: step * (i + 1)], dest=i, tag=TEXT_SEND_TAG)
        text = text[:step]
    else:
        text = comm.recv(source=0, tag=TEXT_SEND_TAG)

    if comm.Get_rank() == 0:
        start_time = time.time()
    bpe_obj.train(text)
    if comm.Get_rank() == 0:
        exec_time = time.time() - start_time
        out_path = os.path.join(cfg.out_folder, f'voc_{cfg.vocab_size}.txt')
        with open(out_path, 'w') as f:
            f.write(str(exec_time))


if __name__ == "__main__":
    main()
