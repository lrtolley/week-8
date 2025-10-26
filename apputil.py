import random
from collections import defaultdict
import numpy as np

class MarkovText:
    def __init__(self, corpus):
        if isinstance(corpus, str):
            self.tokens = corpus.split()
        elif isinstance(corpus, bytes):
            self.tokens = corpus.decode().split()
        else:
            try:
                seq = list(corpus)
            except TypeError:
                raise TypeError("corpus must be a string, bytes, or an iterable of tokens")
            if len(seq) == 1 and isinstance(seq[0], str) and any(ch.isspace() for ch in seq[0]):
                self.tokens = seq[0].split()
            else:
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

    def _random_state_with_followers(self):
        # return a random key that has at least one follower; fallback to any token if none
        keys_with_followers = [k for k, v in self.term_dict.items() if v]
        return random.choice(keys_with_followers) if keys_with_followers else random.choice(self.tokens)

    def generate(self, term_count=20, seed_term=None):
        if self.term_dict is None:
            self.get_term_dict()

        try:
            term_count = int(term_count)
        except (TypeError, ValueError):
            raise TypeError("term_count must be an integer or convertible to int")
        if term_count <= 0:
            return []

        # Validate seed against observed Markov states (term_dict keys)
        if seed_term is None:
            current = self._random_state_with_followers()
        else:
            if seed_term not in self.term_dict:
                raise ValueError("seed_term not found in corpus")
            # If the seed exists but has no followers, start from a state that does
            if not self.term_dict[seed_term]:
                current = self._random_state_with_followers()
            else:
                current = seed_term

        output = [current]
        while len(output) < term_count:
            followers = self.term_dict.get(current, [])
            if followers:
                next_token = np.random.choice(followers)
            else:
                # pick a random state that has followers (keeps transitions valid)
                current = self._random_state_with_followers()
                next_token = np.random.choice(self.term_dict[current])
            output.append(next_token)
            current = next_token

        return output
