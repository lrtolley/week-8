from collections import defaultdict
import random
import numpy as np
import re

class MarkovText:
    def __init__(self, corpus):
        """
        corpus: str containing the whole text corpus
        """
        self.raw_corpus = corpus
        self.tokens = self._tokenize(corpus)
        self.term_dict = self.get_term_dict()

    def _tokenize(self, text):
        """
        Simple tokenizer: splits on whitespace and keeps common punctuation as separate tokens.
        Adjust if your tests expect a different tokenization.
        """
        # separate punctuation from words so punctuation becomes tokens too
        # keep apostrophes inside words
        toks = re.findall(r"[A-Za-z0-9']+|[.,!?;:()\"-]", text)
        return toks

    def get_term_dict(self):
        """
        Builds a dictionary mapping each token to a list of tokens that follow it.
        Duplicates are included to preserve empirical transition probabilities.
        """
        term_dict = defaultdict(list)
        for i in range(len(self.tokens) - 1):
            cur = self.tokens[i]
            nxt = self.tokens[i + 1]
            term_dict[cur].append(nxt)
        return dict(term_dict)

    def generate(self, term_count=20, seed_term=None):
        """
        Generate text of length `term_count` tokens.
        If seed_term is provided but not present in the corpus, raise ValueError.
        If a token has no followers (dead-end), choose a new random token to continue.
        """
        if term_count <= 0:
            return ""

        if seed_term is not None:
            if seed_term not in self.term_dict:
                raise ValueError(f"Seed term '{seed_term}' not found in corpus.")
            current = seed_term
        else:
            # choose a random starting token that has at least one follower
            keys_with_followers = [k for k, v in self.term_dict.items() if v]
            if not keys_with_followers:
                return ""  # no transitions at all
            current = random.choice(keys_with_followers)

        output = [current]

        for _ in range(term_count - 1):
            followers = self.term_dict.get(current)
            if not followers:
                # if no followers, pick a new random key that has followers
                keys_with_followers = [k for k, v in self.term_dict.items() if v]
                if not keys_with_followers:
                    break
                current = random.choice(keys_with_followers)
                output.append(current)
                continue

            current = np.random.choice(followers)
            output.append(current)

        return ' '.join(output)
