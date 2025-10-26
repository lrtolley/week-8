from collections import defaultdict
import random


class MarkovText(object):

    def __init__(self, corpus):
        self.corpus = corpus
        self.term_dict = None  # you'll need to build this

    def get_term_dict(self):

        # build a dict mapping each token to a list of followers (duplicates preserved)
        tokens = self.corpus.split() if self.corpus is not None else []
        d = defaultdict(list)

        # populate followers
        for i in range(len(tokens) - 1):
            d[tokens[i]].append(tokens[i + 1])

        # ensure last token exists with empty follower list (if any tokens)
        if tokens:
            d[tokens[-1]]  # access to create key with empty list if needed

        # convert to regular dict, store and return
        self.term_dict = {k: list(v) for k, v in d.items()}
        return self.term_dict


    def generate(self, seed_term=None, term_count=15):

        # ensure term_dict exists
        if self.term_dict is None:
            self.get_term_dict()

        if term_count <= 0:
            return ""

        if not self.term_dict:
            raise ValueError("Corpus contains no terms to generate from.")

        # choose starting term
        if seed_term is None:
            current = random.choice(list(self.term_dict.keys()))
        else:
            if seed_term not in self.term_dict:
                raise ValueError("seed_term not in term dictionary")
            current = seed_term

        output = [current]

        # generate up to term_count tokens
        while len(output) < term_count:
            followers = self.term_dict.get(current, [])
            if not followers:
                break
            current = random.choice(followers)
            output.append(current)

        return " ".join(output)