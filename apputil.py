import random
from collections import defaultdict
import numpy as np
import pandas as pd
import requests
import re

class MarkovText:
    def __init__(self, corpus):
        self.corpus = corpus.split()  # Tokenize by whitespace
        self.term_dict = self.get_term_dict()

    def get_term_dict(self):
        term_dict = defaultdict(list)
        for i in range(len(self.corpus) - 1):
            current_token = self.corpus[i]
            next_token = self.corpus[i + 1]
            term_dict[current_token].append(next_token)
        return dict(term_dict)

    def _normalize_term_count(self, term_count):
        # Must raise exact TypeError text when not convertible
        if isinstance(term_count, bool):
            raise TypeError("term_count must be an integer or convertible to int")
        if isinstance(term_count, int):
            return int(term_count)
        # try direct conversion (handles strings like "10" and numeric types)
        try:
            return int(term_count)
        except Exception:
            pass
        # if it's a single-element iterable (e.g., [10], (10,), array-like), try extracting the single element
        try:
            seq = list(term_count)
        except Exception:
            raise TypeError("term_count must be an integer or convertible to int")
        else:
            if len(seq) == 1:
                try:
                    return int(seq[0])
                except Exception:
                    raise TypeError("term_count must be an integer or convertible to int")
        raise TypeError("term_count must be an integer or convertible to int")

    def _random_state_with_followers(self):
        if self.term_dict is None:
            self.get_term_dict()
        keys_with = [k for k, v in self.term_dict.items() if v]
        if keys_with:
            return random.choice(keys_with)
        if self.tokens:
            return random.choice(self.tokens)
        return None

        def generate(self, term_count=20, seed_term=None):
        if seed_term:
            if seed_term not in self.term_dict:
                raise ValueError(f"Seed term '{seed_term}' not found in corpus.")
            current_term = seed_term
        else:
            current_term = random.choice(list(self.term_dict.keys()))

        output = [current_term]

        for _ in range(term_count - 1):
            next_terms = self.term_dict.get(current_term)
            if not next_terms:
                # If no next term, pick a new random seed
                current_term = random.choice(list(self.term_dict.keys()))
                output.append(current_term)
                continue
            current_term = np.random.choice(next_terms)
            output.append(current_term)

        return ' '.join(output)
