import re
from collections import Counter, defaultdict

from tqdm import tqdm


class Token:
    left: bytes | None
    right: bytes | None
    val: bytes

    def __init__(self, left: bytes | None, right: bytes | None, val: bytes):
        self.left = left
        self.right = right
        self.val = val

    def __hash__(self) -> int:
        return int.from_bytes(self.val, "little")

    def __eq__(self, other: object) -> bool:
        return self.__hash__() == other.__hash__()

    def __repr__(self) -> str:
        return f"Token({self.left} + {self.right} -> {self.val})"


class Tokeniser:
    vocab_size: int
    corpus: list[list[bytes]]

    def __init__(self, vocab_size=128):
        self.vocab_size = vocab_size
        self.corpus = []

    def pre_tokenise(self, text: str) -> list[list[bytes]]:
        pattern = (
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\s\d]+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""
        )
        words = [w.encode("utf-8") for w in re.findall(pattern, text)]
        return [[bytes([b]) for b in w] for w in words]

    def fit(self, text: str):
        self.corpus = self.pre_tokenise(text)

        # initialise vocab with individual characters
        self.vocab = Counter()
        for word in tqdm(self.corpus):
            tokenised
            for char in word:
                token = Token(None, None, char)
                self.vocab[token] += 1

        pbar = tqdm(
            total=self.vocab_size, desc="Building vocab", initial=len(self.vocab)
        )
        while len(self.vocab) < self.vocab_size:
            most_common = self.most_common_pair()
            self.merge_pairs(most_common)
            pbar.update(1)

        self.tokens = {k: i for i, k in enumerate(self.vocab)}

    def most_common_pair(self):
        pairs = Counter()
        for word in self.corpus:
            if len(word) in {0, 1}:
                continue
            for left, right in zip(word[:-1], word[1:]):
                token = Token(left, right, left + right)
                pairs[token] += 1

        return pairs.most_common(1)[0][0]

    def merge_pairs(self, most_common: Token):
        for word_idx, word in enumerate(self.corpus):
            if len(word) in {0, 1}:
                continue

            i = 0
            while i < len(word) - 1:
                left = word[i]
                right = word[i + 1]
                if left == most_common.left and right == most_common.right:
                    self.vocab[left] -= 1
                    self.vocab[right] -= 1
                    word = word[:i] + [most_common.val] + word[i + 2 :]
                    self.vocab[most_common] += 1

                i += 1
            self.corpus[word_idx] = word

    def tokenise(self, text: str) -> list[str]:
        tokens = []
        for word in self.pre_tokenise(text):
            for token in list(self.tokens.keys())[::-1]:
                pass


if __name__ == "__main__":
    with open("/Users/benedictclarkson/Downloads/bee_movie.txt", "r") as f:
        text = f.read()

    t = Tokeniser(256)
    t.fit(text)
    print(t.tokens)

    # print(text[:100])
    # print(t.pre_tokenise(text)[:100])

    # print(text[:100])
    # print(t.tokenise(text[:100]))
