import hydra
from mpi4py import MPI
from omegaconf import DictConfig

import bpe

TEXT_SEND_TAG = 1

@hydra.main(config_name="config")
def main(cfg: DictConfig):
    comm = MPI.COMM_WORLD

    bpe_obj = bpe.BPE(max_iters=cfg.max_iters,
                      vocab_size=cfg.vocab_size,
                      tokens_path=hydra.utils.to_absolute_path(cfg.tokens_path),
                      id2token_path=hydra.utils.to_absolute_path(cfg.id2token_path),
                      encodings_path=hydra.utils.to_absolute_path(cfg.encodings_path),
                      comm=comm)

    with open(hydra.utils.to_absolute_path(cfg.encode_text_path), "r") as f:
        text = f.read()

    bpe_obj.encode(text.split("\n"))


if __name__ == "__main__":
    main()
