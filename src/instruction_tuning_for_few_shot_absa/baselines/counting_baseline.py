from typing import List
import datasets
import nltk

x = datasets.load_dataset("amazon_us_reviews", "Wireless_v1_00")


def keep_only_nouns(sentence) -> List[str]:
    [x[0] for x in nltk.pos_tag(nltk.word_tokenize(sentence)) if "NN" in x[1]]


ap["train"].map(
    lambda x: {
        "title_only_nouns": " ".join([token.text for token in nlp(x["title"])]),
        "content_only_nouns": " ".join([token.text for token in nlp(x["title"])]),
    },
    batched=False,
)
