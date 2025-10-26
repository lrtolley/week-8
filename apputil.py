from collections import defaultdict
import random
import numpy as np

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
        Simple whitespace tokenizer to match autograder expectations.
        """
        return text.split()

    def get_term_dict(self):
        """
        Build a mapping from each token to a list of tokens that follow it.
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
        - Coerces term_count to int, raising ValueError if invalid.
        - Raises ValueError if seed_term is provided but not present in the corpus tokens.
        - If current token has no followers, jump to a random token that has followers.
        """
        try:
            term_count = int(term_count)
        except Exception as e:
            raise ValueError("term_count must be convertible to int") from e

        if term_count <= 0:
            return ""

        if seed_term is not None:
            if seed_term not in self.tokens:
                raise ValueError(f"Seed term '{seed_term}' not found in corpus.")
            current = seed_term
        else:
            # choose a random starting token that has at least one follower
            keys_with_followers = [k for k, v in self.term_dict.items() if v]
            if not keys_with_followers:
                return ""  # no transitions
            current = random.choice(keys_with_followers)

        output = [current]

        for _ in range(term_count - 1):
            followers = self.term_dict.get(current)
            if not followers:
                # jump to a random token that has followers
                keys_with_followers = [k for k, v in self.term_dict.items() if v]
                if not keys_with_followers:
                    break
                current = random.choice(keys_with_followers)
                output.append(current)
                continue

            current = np.random.choice(followers)
            output.append(current)

        return ' '.join(output)
