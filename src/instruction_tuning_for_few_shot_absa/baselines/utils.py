from typing import List
from spacy.language import Language
import re
import string


# function to remove non-ascii characters
def remove_non_ascii(s: str):
    return "".join(i for i in s if ord(i) < 128)


"""
Lemmatize s and return a list with the lemmas
"""


def lemmatize(s: str, nlp: Language) -> List[str]:
    doc = nlp(s, disable=["parser", "ner"])
    return [token.lemma_.lower() for token in doc]


"""
Removes punctuation and multiple spaces
"""


def remove_punctuation(s: str) -> str:
    regex = re.compile("[" + re.escape(string.punctuation) + "\\r\\t\\n]")
    no_punct = regex.sub(" ", str(s))
    result = " ".join(no_punct.split())
    return result


"""
Some reviews contain some html tags
This function removes them (multiple calls to replace)
"""


def simple_html_tags_remover(s: str) -> str:
    return s.replace("<br>", "").replace("</br>", "").replace("<br />", "")


def clean_comments(s: str, nlp: Language):
    x = simple_html_tags_remover(s)
    x = remove_non_ascii(x)
    x = remove_punctuation(x)
    x = lemmatize(x, nlp)
    x = [w for w in x if "u00" not in w]  # remove non utf 8 words?

    return x


def map_clean_comments(ss: List[str], nlp: Language) -> List[str]:
    return [clean_comments(s, nlp) for s in ss]
