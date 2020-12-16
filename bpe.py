import pickle
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

from tqdm.auto import tqdm


class ModelNotTrainedError(Exception):
    pass


class BPE:
    """
    Class for training and applying Byte Pair Encoding
    :param vocab_size: resulting vocabulary size
    :param max_iters: maximum iterations to perform. By default vocab_size * 2
    :param bpe_patj: path to file to load vocab from and save vocab to
    :param comm: MPI.COMM_WORLD object for parallel processing
    """
    def __init__(self, vocab_size: int, max_iters: Optional[int] = None, bpe_path: str = None, comm=None) -> None:
        self.vocab_size = vocab_size
        self.max_iters = max_iters if max_iters is not None else vocab_size * 2
        self.vocab = Counter()
        self.comm = comm
        if bpe_path is not None:
            self.bpe_path = Path(bpe_path)
            if self.bpe_path.exists():
                with open(self.bpe_path, "rb") as f:
                    self.vocab = pickle.load(f)
                    assert isinstance(self.vocab, Counter)

    def train(self, corpus: str) -> None:
        corpus = corpus.lower().split()

        # Take only necessary for this process part of corpus
        step = len(corpus) // self.comm.Get_size() + 1
        corpus = corpus[step * self.comm.Get_rank(): step * (self.comm.Get_rank() + 1)]
        vocab = Counter(c for word in corpus for c in word)  # Count each element
        vocab["</w>"] = len(corpus)
        vocab = self.comm.allreduce(vocab)

        # Transform words into tuples and count the occurrences of each word
        corpus = [tuple(word) + ("</w>",) for word in corpus]
        corpus = Counter(corpus)

        if self.comm.Get_rank() == 0:
            pbar = tqdm(total=self.vocab_size)  # Progress bar
        for _ in range(self.max_iters):
            # Update progress bar
            if self.comm.Get_rank() == 0:
                pbar.update(len(vocab) - pbar.n)
            if len(vocab) >= self.vocab_size:
                break

            # Count pairs' frequencies
            pairs = Counter()
            for word, freq in corpus.items():
                for pair in zip(word[:-1], word[1:]):
                    pairs[pair] += freq
            pairs = self.comm.allreduce(pairs)
            most_common_pair, most_common_pair_freq = pairs.most_common(1)[0]
            most_common_pair_str = most_common_pair[0] + most_common_pair[1]

            # Merge most frequent pair in corpus
            new_corpus = Counter()
            for word in corpus:
                new_word = list(word)
                for i in range(len(word) - 1):
                    if word[i] == most_common_pair[0] and word[i + 1] == most_common_pair[1]:
                        new_word[i] = most_common_pair_str
                        new_word[i + 1] = None

                new_word = tuple(c for c in new_word if c is not None)
                new_corpus[new_word] = corpus[word]
            corpus = new_corpus

            # Add new pair to vocab and subtract number of occurrences of its parts
            vocab[most_common_pair_str] += most_common_pair_freq
            vocab[most_common_pair[0]] -= most_common_pair_freq
            vocab[most_common_pair[1]] -= most_common_pair_freq
            if vocab[most_common_pair[0]] <= 0:
                del vocab[most_common_pair[0]]
            if vocab[most_common_pair[1]] <= 0:
                del vocab[most_common_pair[1]]
        else:  # Maximum iterations exceeded
            print("Warning: Maximum iterations exceeded. Consider lowering vocab_size")
        if self.comm.Get_rank() == 0:
            pbar.close()

        # Save vocab
        self.vocab = vocab
        if self.comm.Get_rank() == 0:
            with open(self.bpe_path, "wb") as f:
                pickle.dump(self.vocab, f)

    def encode(self, corpus: str) -> List[str]:
        """
        Apply byte pair encodung to text
        :param corpus: string with text to encode
        :return: list of strings -- encoded text
        """
        if len(self.vocab) == 0:
            raise ModelNotTrainedError("BPE model is not trained. Call train before applying the model")
        raise NotImplementedError()
