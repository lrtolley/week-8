import random
from collections import defaultdict

class MarkovText:
    def __init__(self, corpus):
        """
        Robust corpus handling:
        - str -> split on whitespace
        - bytes -> decode then split
        - iterable of tokens -> use items as tokens
        - single-element iterable whose element is a str -> split that inner string
        This initializer never calls .split on a list object itself.
        """
        if isinstance(corpus, str):
            self.tokens = corpus.split()
        elif isinstance(corpus, bytes):
            self.tokens = corpus.decode().split()
        else:
            try:
                outer = list(corpus)
            except TypeError:
                raise TypeError("corpus must be a string, bytes, or an iterable of tokens")
            if len(outer) == 0:
                self.tokens = []
            elif len(outer) == 1 and isinstance(outer[0], str):
                self.tokens = outer[0].split()
            elif len(outer) == 1 and isinstance(outer[0], bytes):
                self.tokens = outer[0].decode().split()
            elif len(outer) == 1:
                try:
                    inner = list(outer[0])
                except TypeError:
                    self.tokens = [str(outer[0])]
                else:
                    processed = []
                    for it in inner:
                        if isinstance(it, bytes):
                            processed.append(it.decode())
                        else:
                            processed.append(str(it))
                    self.tokens = processed
            else:
                processed = []
                for it in outer:
                    if isinstance(it, bytes):
                        processed.append(it.decode())
                    else:
                        processed.append(str(it))
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
        if self.term_dict is None:
            self.get_term_dict()

        term_count = self._normalize_term_count(term_count)
        if term_count <= 0:
            return []

        # seed must be a seen state (key in term_dict)
        if seed_term is None:
            current = self._random_state_with_followers()
            if current is None:
                return []
        else:
            if seed_term not in self.term_dict:
                raise ValueError("seed_term not found in corpus")
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
                nxt = random.choice(followers)
            else:
                current = self._random_state_with_followers()
                followers = self.term_dict.get(current, [])
                if followers:
                    nxt = random.choice(followers)
                else:
                    nxt = random.choice(self.tokens) if self.tokens else None
            if nxt is None:
                break
            output.append(nxt)
            current = nxt

        return output
