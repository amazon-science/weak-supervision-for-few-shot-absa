"""
Output statistics of how the aspect words look like
- number of words
- frequency (in their dataset)
- frequency (in amazon_us_reviews)
"""

from ast import literal_eval
from collections import defaultdict
import spacy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def number_of_words(path, save_path, log_words_path):
    fout = open(log_words_path, "w+")
    aspect_terms = []
    with open(path) as fin:
        for line in fin:
            absa_quads = literal_eval(line.split("####")[-1])
            for quad in absa_quads:
                aspect_terms.append(quad[0])

    nlp = spacy.load("en_core_web_sm")
    counts_with_null = defaultdict(int)
    counts_without_null = defaultdict(int)
    for at in aspect_terms:
        doc = nlp(at)
        split = [x.text for x in doc]
        counts_with_null[len(split)] += 1
        if len(split) > 1:
            fout.write(
                f"{at}\t{len(split)}\t{at in [x.text for x in doc.noun_chunks]}\n"
            )
        if at != "NULL":
            counts_without_null[len(split)] += 1

    fout.close()

    c_with_n = sorted(counts_with_null.items(), key=lambda x: x[0])
    c_without_n = sorted(counts_without_null.items(), key=lambda x: x[0])

    ax = sns.barplot(
        x=[x[0] for x in c_without_n], y=[x[1] for x in c_without_n], color="dodgerblue"
    )
    ax.set_yscale("log")
    plt.xlabel("# of words")
    plt.ylabel("count (log)")
    plt.savefig(save_path)


rest15 = "data/rest15/train.txt"
rest16 = "data/rest16/train.txt"

number_of_words(
    rest15,
    "results/week2/aspect_terms_extraction/rest15_number_of_words.png",
    "results/week2/aspect_terms_extraction/rest15_long_at",
)
number_of_words(
    rest16,
    "results/week2/aspect_terms_extraction/rest16_number_of_words.png",
    "results/week2/aspect_terms_extraction/rest16_long_at",
)
