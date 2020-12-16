import pickle
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

import hydra
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
    :param verbose: wether to print progress bar
    """
    def __init__(self, vocab_size: int, max_iters: Optional[int] = None, tokens_path: str = None,
                 id2token_path: str = None, encodings_path: str = None, comm=None, verbose: bool = True) -> None:
        self.vocab_size = vocab_size
        self.max_iters = max_iters if max_iters is not None else vocab_size * 2
        self.vocab = Counter()
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        self.verbose = verbose
        if tokens_path is not None:
            self.tokens_path = Path(tokens_path)
            if self.tokens_path.exists():
                with open(self.tokens_path, "rb") as f:
                    self.tokens = pickle.load(f)
                    assert isinstance(self.tokens, list)
                    
        if id2token_path is not None:
            self.id2token_path = Path(id2token_path)
            if self.id2token_path.exists():
                with open(self.id2token_path, "rb") as f:
                    self.id2token = pickle.load(f)
                    assert isinstance(self.id2token, Counter)
                    
        if encodings_path is not None:
            self.encodings_path = Path(encodings_path)

    def train(self, corpus: str) -> None:
        rank = self.comm.Get_rank()

        corpus = corpus.lower().split()
        vocab = Counter(c for word in corpus for c in word)  # Count each element
        vocab["</w>"] = len(corpus)
        vocab = self.comm.allreduce(vocab)

        # Transform words into tuples and count the occurrences of each word
        # leave only the part necessary for this process
        corpus = [tuple(word) + ("</w>",) for word in corpus]
        corpus = Counter(corpus)
        corpus = self.comm.allreduce(corpus)
        step = len(corpus) // self.comm.Get_size() + 1
        corpus = dict(list(corpus.items())[step * rank: step * (rank + 1)])

        if rank == 0 and self.verbose:
            pbar = tqdm(total=self.vocab_size)  # Progress bar
        for _ in range(self.max_iters):
            # Update progress bar
            if rank == 0 and self.verbose:
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
        if rank == 0 and self.verbose:
            pbar.close()

        tokens_sorted = sorted(vocab.keys(), 
                               key=lambda tok: len(tok) - 3 if tok.endswith('</w>') else len(tok),
                               reverse=True)
        self.tokens = tokens_sorted
        self.id2token = {i + 1: tok for i, tok in enumerate(tokens_sorted)}
        self.id2token[0] = '<UNK>'
        # Save vocab
        if rank == 0:
            with open(hydra.utils.to_absolute_path(self.tokens_path), "wb") as f:
                pickle.dump(self.tokens, f)
            with open(hydra.utils.to_absolute_path(self.id2token_path), "wb") as f:
                pickle.dump(self.id2token, f)

    def encode(self, corpus: str) -> List[int]:
        """
        Apply byte pair encodung to text
        :param corpus: string with text to encode
        :return: list of strings -- encoded text
        """
        if len(self.tokens) == 0:
            raise ModelNotTrainedError("BPE model is not trained. Call train before applying the model")
        
        words = corpus.strip().split()
        words_per_process = len(words) // size

        if self.rank == self.size - 1:
            words = words[self.rank * words_per_process:]
        else:
            words = words[self.rank * words_per_process:(rank + 1) * words_per_process]

        words_string = '</w>'.join(words) + '</w>'

        def _encode(string, tokens, id):
            if string == '':
                return []
            if len(tokens) == 0:
                return [0]

            token = token[0]
            token_reg = re.escape(token)

            string_tokens = []
            matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
            if len(matched_positions) == 0:
                return _encode(string, tokens[1:], id + 1)
            substring_end_positions = [matched_position[0] for matched_position in matched_positions]

            substring_start_position = 0
            for substring_end_position in substring_end_positions:
                substring = string[substring_start_position:substring_end_position]
                string_tokens += _encode(substring, tokens[1:], id + 1)
                string_tokens += [id]
                substring_start_position = substring_end_position + len(token)
            remaining_substring = string[substring_start_position:]
            string_tokens += _encode(remaining_substring, tokens[1:], id + 1)

            return string_tokens

        ids = _encode(words_string, self.tokens, 1)
        ids = self.comm.gather(ids, root=0)
        ids = [item for sublist in t for item in sublist]
        
        with open(hydra.utils.to_absolute_path(self.encodings_path), "wb") as f:
                pickle.dump(ids, f)
    
        
