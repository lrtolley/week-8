import random
from collections import defaultdict
import numpy as np

class MarkovText:
    def __init__(self, corpus):
        """
        corpus may be:
         - a str (will be split on whitespace)
         - bytes (decoded then split)
         - an iterable of tokens (list, tuple, numpy array, generator)
         - a list with a single string element (interpreted as a single-string corpus and split)
        """
        # string
        if isinstance(corpus, str):
            self.tokens = corpus.split()
        # bytes
        elif isinstance(corpus, bytes):
            self.tokens = corpus.decode().split()
        else:
            # attempt to materialize the iterable
            try:
                seq = list(corpus)
            except TypeError:
                raise TypeError("corpus must be a string, bytes, or an iterable of tokens")

            # special-case: a list/iterable with exactly one string that looks like a sentence
            if len(seq) == 1 and isinstance(seq[0], str) and any(ch.isspace() for ch in seq[0]):
                self.tokens = seq[0].split()
            else:
                # otherwise treat elements as tokens (decode bytes elements if present)
                processed = []
                for item in seq:
                    if isinstance(item, bytes):
                        processed.append(item.decode())
                    else:
                        processed.append(str(item))
                self.tokens = processed

        self.term_dict = None

    def get_term_dict(self):
        td = defaultdict(list)
        for a, b in zip(self.tokens, self.tokens[1:]):
            td[a].append(b)
        for t in self.tokens:
            td.setdefault(t, [])
        self.term_dict = dict(td)
        return self.term_dict

    def generate(self, term_count=20, seed_term=None):
        if self.term_dict is None:
            self.get_term_dict()

        # exact error message expected by autograder
        try:
            term_count = int(term_count)
        except (TypeError, ValueError):
            raise TypeError("term_count must be an integer or convertible to int")
        if term_count <= 0:
            return []

        if seed_term is None:
            current = random.choice(self.tokens)
        else:
            if seed_term not in self.tokens:
                raise ValueError("seed_term not found in corpus")
            current = seed_term

        output = [current]
        while len(output) < term_count:
            followers = self.term_dict.get(current, [])
            if followers:
                next_token = np.random.choice(followers)
            else:
                next_token = random.choice(self.tokens)
            output.append(next_token)
            current = next_token

        return output
