import random
from collections import defaultdict
import numpy as np

class MarkovText:
    def __init__(self, corpus):
        """
        corpus: either a single string or an iterable of tokens.
        """
        # Accept either a single string or any iterable of tokens
        if isinstance(corpus, str):
            self.tokens = corpus.split()
        else:
            # Defensive: if corpus is bytes, decode first; otherwise convert iterable to list
            if isinstance(corpus, bytes):
                self.tokens = corpus.decode().split()
            else:
                self.tokens = list(corpus)
        self.term_dict = None

    def get_term_dict(self):
        """
        Build a dict where each token maps to a list of following tokens.
        Duplicates are kept so sampling reflects observed frequencies.
        """
        td = defaultdict(list)
        for a, b in zip(self.tokens, self.tokens[1:]):
            td[a].append(b)
        for t in self.tokens:
            td.setdefault(t, [])
        self.term_dict = dict(td)
        return self.term_dict

    def generate(self, term_count=20, seed_term=None):
        """
        Generate exactly term_count tokens.
        term_count is validated and converted to int if possible.
        seed_term must be an element of the token list or None.
        If the current token has no followers, a random token from the corpus is chosen.
        """
        if self.term_dict is None:
            self.get_term_dict()

        # Validate and normalize term_count
        try:
            term_count = int(term_count)
        except Exception:
            raise TypeError("term_count must be an integer or convertible to int")
        if term_count <= 0:
            return []

        # Validate seed_term
        if seed_term is None:
            current = random.choice(self.tokens)
        else:
            # Accept seed_term as any hashable token; membership tested against tokens list
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
