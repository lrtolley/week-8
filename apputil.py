import random
from collections import defaultdict
import numpy as np

class MarkovText:
    def __init__(self, corpus):
        """
        corpus: iterable of tokens (list of strings) or a single string which will be split on whitespace.
        """
        if isinstance(corpus, str):
            self.tokens = corpus.split()
        else:
            self.tokens = list(corpus)
        self.term_dict = None

    def get_term_dict(self):
        """
        Build and return a dict where each key is a token and the value is a list
        of tokens that follow that key in the corpus. Duplicates are kept so next-token
        sampling reflects observed frequencies.
        """
        td = defaultdict(list)
        for a, b in zip(self.tokens, self.tokens[1:]):
            td[a].append(b)
        # ensure every token appears as a key (even if it has no followers)
        for t in self.tokens:
            td.setdefault(t, [])
        self.term_dict = dict(td)
        return self.term_dict

    def generate(self, term_count=20, seed_term=None):
        """
        Generate up to term_count tokens using the 1-word Markov property.
        If seed_term is provided but not found in the corpus, raise ValueError.
        If a current token has no followers, generation stops early and the current sequence is returned.
        """
        if self.term_dict is None:
            self.get_term_dict()

        if term_count <= 0:
            return []

        if seed_term is None:
            current = random.choice(self.tokens)
        else:
            if seed_term not in self.term_dict:
                raise ValueError("seed_term not found in corpus")
            current = seed_term

        output = [current]
        for _ in range(term_count - 1):
            followers = self.term_dict.get(current, [])
            if not followers:
                break
            next_token = np.random.choice(followers)
            output.append(next_token)
            current = next_token

        return output
