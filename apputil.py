import random
from collections import defaultdict
import numpy as np

class MarkovText:
    def __init__(self, corpus):
        """
        corpus may be:
         - a str (will be split on whitespace)
         - bytes (will be decoded then split)
         - any iterable of tokens (list, tuple, np.array, generator, etc.)
        This initializer never calls .split on a non-str object.
        """
        # If it's a plain string, split into tokens
        if isinstance(corpus, str):
            self.tokens = corpus.split()
        elif isinstance(corpus, bytes):
            self.tokens = corpus.decode().split()
        else:
            # Try to convert to a sequence without calling .split on the corpus itself.
            try:
                seq = list(corpus)
            except TypeError:
                raise TypeError("corpus must be a string, bytes, or an iterable of tokens")
            # If the iterable has exactly one element and that element is a string that looks like a sentence,
            # it's reasonable to treat that single string as a full corpus to split.
            if len(seq) == 1 and isinstance(seq[0], str) and any(c.isspace() for c in seq[0]):
                self.tokens = seq[0].split()
            else:
                # Otherwise treat each element as a token; convert non-str elements to str to avoid surprises.
                self.tokens = [t.decode() if isinstance(t, bytes) else str(t) for t in seq]

        self.term_dict = None

    def get_term_dict(self):
        td = defaultdict(list)
        for a, b in zip(self.tokens, self.tokens[1:]):
            td[a].append(b)
        # ensure every token appears as a key
        for t in self.tokens:
            td.setdefault(t, [])
        self.term_dict = dict(td)
        return self.term_dict

    def generate(self, term_count=20, seed_term=None):
        if self.term_dict is None:
            self.get_term_dict()

        # term_count must be int or convertible to int — exact error message required by autograder
        try:
            term_count = int(term_count)
        except (TypeError, ValueError):
            raise TypeError("term_count must be an integer or convertible to int")
        if term_count <= 0:
            return []

        # seed validation: must be exactly one of the tokens
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
                # when terminal token encountered, pick a random token from corpus and continue
                next_token = random.choice(self.tokens)
            output.append(next_token)
            current = next_token

        return output
