import random
from collections import defaultdict
import numpy as np

class MarkovText:
    def __init__(self, corpus):
        """
        corpus: a single string, bytes, or an iterable of tokens (commonly a list of strings).
        Behavior:
        - If corpus is a str: split on whitespace.
        - If corpus is bytes: decode then split on whitespace.
        - If corpus is a list with a single string element: split that string.
        - Otherwise treat corpus as an iterable of tokens and convert each item to str.
        """
        # String or bytes: split into tokens
        if isinstance(corpus, str):
            self.tokens = corpus.split()
        elif isinstance(corpus, bytes):
            self.tokens = corpus.decode().split()
        else:
            # Iterable case: explicitly handle single-item list containing a full string
            try:
                seq = list(corpus)
            except TypeError:
                raise TypeError("corpus must be a string, bytes, or an iterable of tokens")
            if len(seq) == 1 and isinstance(seq[0], str) and (" " in seq[0] or "\n" in seq[0] or "\t" in seq[0]):
                # a list like ["word1 word2 ..."] - interpret as a single string corpus to split
                self.tokens = seq[0].split()
            else:
                # Otherwise treat each element as a token (convert to str to be safe)
                self.tokens = [str(t) for t in seq]

        self.term_dict = None

    def get_term_dict(self):
        """
        Build and return a dict mapping token -> list of followers (duplicates kept).
        Ensures every observed token appears as a key (possibly with an empty list).
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
        - term_count must be an int or convertible to int; otherwise raise TypeError with the exact message.
        - If seed_term is provided but not present in the token list, raise ValueError.
        - If the current token has no followers, pick a random token from the corpus and continue.
        Returns a list of tokens of length term_count (or [] if term_count <= 0).
        """
        if self.term_dict is None:
            self.get_term_dict()

        # Validate and normalize term_count
        try:
            term_count = int(term_count)
        except (TypeError, ValueError):
            raise TypeError("term_count must be an integer or convertible to int")
        if term_count <= 0:
            return []

        # Validate seed_term membership
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
