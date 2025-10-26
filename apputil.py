import random
from collections import defaultdict
import numpy as np

class MarkovText:
    def __init__(self, corpus):
        """
        Accepts:
         - str -> split on whitespace
         - bytes -> decode then split
         - iterable of tokens (list, tuple, np.array, generator)
         - an iterable with a single element which itself may be:
            * a str -> split that string
            * an iterable of tokens (list/tuple/np.array) -> treat that inner iterable as tokens
        Never calls .split on a list object.
        """
        # direct string/bytes
        if isinstance(corpus, str):
            self.tokens = corpus.split()
        elif isinstance(corpus, bytes):
            self.tokens = corpus.decode().split()
        else:
            # materialize outer iterable
            try:
                seq = list(corpus)
            except TypeError:
                raise TypeError("corpus must be a string, bytes, or an iterable of tokens")

            # if outer iterable is empty
            if len(seq) == 0:
                self.tokens = []
            # single-element outer iterable
            elif len(seq) == 1:
                inner = seq[0]
                # single string inside: split it
                if isinstance(inner, str):
                    self.tokens = inner.split()
                # single bytes inside: decode then split
                elif isinstance(inner, bytes):
                    self.tokens = inner.decode().split()
                # single iterable inside (list/tuple/np.ndarray/generator) -> use its items as tokens
                else:
                    try:
                        inner_seq = list(inner)
                    except TypeError:
                        # not iterable, coerce to str token
                        self.tokens = [str(inner)]
                    else:
                        # convert bytes elements and non-str elements safely
                        processed = []
                        for item in inner_seq:
                            if isinstance(item, bytes):
                                processed.append(item.decode())
                            else:
                                processed.append(str(item))
                        self.tokens = processed
            # multi-element outer iterable: treat each element as a token
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

    def _normalize_term_count(self, term_count):
        # Allow int-like, numpy scalars, or single-element iterables like [10] or np.array([10])
        if isinstance(term_count, (int, np.integer)):
            return int(term_count)
        # If it's a one-element iterable, extract that element
        try:
            seq = list(term_count)
        except TypeError:
            # not iterable and not int-like -> try converting directly
            try:
                return int(term_count)
            except Exception:
                raise TypeError("term_count must be an integer or convertible to int")
        else:
            if len(seq) == 1:
                try:
                    return int(seq[0])
                except Exception:
                    raise TypeError("term_count must be an integer or convertible to int")
            # otherwise not convertible
            raise TypeError("term_count must be an integer or convertible to int")

    def _random_state_with_followers(self):
        if self.term_dict is None:
            self.get_term_dict()
        keys_with_followers = [k for k, v in self.term_dict.items() if v]
        return random.choice(keys_with_followers) if keys_with_followers else random.choice(self.tokens) if self.tokens else None

    def generate(self, term_count=20, seed_term=None):
        if self.term_dict is None:
            self.get_term_dict()

        term_count = self._normalize_term_count(term_count)
        if term_count <= 0:
            return []

        # seed must be an observed state (a key in term_dict)
        if seed_term is None:
            current = self._random_state_with_followers()
            if current is None:
                return []
        else:
            if seed_term not in self.term_dict:
                raise ValueError("seed_term not found in corpus")
            # if seed has no followers, restart from a random state that does
            if not self.term_dict[seed_term]:
                current = self._random_state_with_followers()
                if current is None:
                    return []
            else:
                current = seed_term

        output = [current]
        while len(output) < term_count:
            followers = self.term_dict.get(current, [])
            if followers:
                next_token = np.random.choice(followers)
            else:
                # jump to a random valid state (with followers) and take one of its followers
                current = self._random_state_with_followers()
                followers = self.term_dict.get(current, [])
                if not followers:
                    # no followers anywhere, fallback to random token
                    next_token = random.choice(self.tokens) if self.tokens else None
                else:
                    next_token = np.random.choice(followers)
            if next_token is None:
                break
            output.append(next_token)
            current = next_token

        return output
