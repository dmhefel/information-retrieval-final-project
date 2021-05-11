from typing import Any, List

from nltk.tokenize import word_tokenize  # type: ignore
from nltk.stem.porter import PorterStemmer  # type: ignore
from nltk.corpus import stopwords  # type: ignore
import re
import math

class TextProcessing:
    def __init__(self, stemmer, stop_words, *args):
        """
        class TextProcessing is used to tokenize and normalize tokens that will be further used to build inverted index.
        :param stemmer:
        :param stop_words:
        :param args:
        """
        self.stemmer = stemmer
        self.STOP_WORDS = stop_words

    @classmethod
    def from_nltk(
            cls, stemmer: Any = PorterStemmer().stem, stop_words=None
    ) -> "TextProcessing":
        if stop_words is None:
            stop_words = set(stopwords.words("english"))
        return cls(stemmer, stop_words)

    def is_stop_words(self, token: str) -> bool:
        return token in self.STOP_WORDS

    def is_valid(self, token: str) -> bool:
        return len(token) > 1 and (not self.is_stop_words(token))

    def normalize(self, token: str, use_stemmer: bool) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9\-]", "", token.lower())
        if self.is_valid(normalized):
            if use_stemmer:
                return self.stemmer(normalized)
            else:
                return normalized
        else:
            return ""

    def get_valid_tokens(
            self, title: str, content: str, *, use_stemmer: bool = True
    ) -> List[str]:
        tokens = word_tokenize(content.lower()) + title.lower().split()
        normalized = []
        for tok in tokens:
            normalized_tok = self.normalize(tok, use_stemmer)
            if normalized_tok:
                normalized.append(normalized_tok)

        return normalized

    @staticmethod
    def idf(N: int, df: int) -> float:
        """
        compute the logarithmic (base 2) idf score
        :param N: document count N
        :param df: document frequency
        :return:
        """
        return math.log10(N/df)

    @staticmethod
    def tf(freq: int) -> float:
        """
        compute the logarithmic tf (base 2) score
        :param freq: raw term frequency
        :return:
        """
        if freq>0:
            return 1+ math.log10(freq)
        else:
            return 0.0


if __name__ == "__main__":
    pass
