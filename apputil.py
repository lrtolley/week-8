import random
from collections import defaultdict
import numpy as np

class MarkovText:
    def __init__(self, corpus):
        """
        corpus: either a single string (split on whitespace) or an iterable of tokens.
        """
        if isinstance(corpus, str):
            self.tokens = corpus.split()
        else:
            self.tokens = list(corpus)
        self.term_dict = None

    def get_term_dict(self):
        """
        Build and return a dict where each key is a token and the value is a list
        of tokens that follow that key in the corpus. Duplicates are kept so sampling
        reflects empirical frequencies.
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
        Generate exactly term_count tokens (unless term_count <= 0 returns []).
        If seed_term is provided but not found in the corpus tokens, raise ValueError.
        If the current token has no followers, pick a random token from the corpus
        and continue generation until term_count tokens are produced.
        """
        if self.term_dict is None:
            self.get_term_dict()

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
                # When no followers for the current token, pick a random token from corpus
                next_token = random.choice(self.tokens)
            output.append(next_token)
            current = next_token

        return output
