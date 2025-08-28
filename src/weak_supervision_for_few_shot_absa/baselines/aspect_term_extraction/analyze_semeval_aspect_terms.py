"""
Analyze how the aspect terms in SemEval look like
POS tag frequency for multi-word aspect terms
"""

from dataclasses import dataclass
from typing import Dict, List
import nltk
import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.weak_supervision_for_few_shot_absa.baselines.aspect_term_extraction.most_frequent_nouns import (
    build_ngrams,
    map_clean_comments,
)


@dataclass
class ABSAQuad:
    aspect_term: str
    aspect_category: str
    sentiment: str
    opinion_term: str


@dataclass
class ABSAExample:
    sentence: List[str]
    quads: List[ABSAQuad]


def read_semeval_format(path) -> List[ABSAExample]:
    result: List[ABSAExample] = []
    with open(path) as fin:
        for line in fin:
            sentence, tuples = line.split("####")
            labels = eval(tuples)
            quads: List[ABSAQuad] = []
            for l in labels:
                quads.append(
                    ABSAQuad(
                        aspect_term=l[0],
                        aspect_category=l[1],
                        sentiment=l[2],
                        opinion_term=l[3],
                    )
                )
            result.append(ABSAExample(sentence.split(" "), quads))
    return result


def analyze_pos_tags():
    lap14 = read_semeval_format(
        "<ABSAWithWeaklySupervisedPreTraining_Path>/data/lap14/train.txt"
    )
    rest15 = read_semeval_format(
        "<ABSAWithWeaklySupervisedPreTraining_Path>/data/rest15/train.txt"
    )
    rest16 = read_semeval_format(
        "<ABSAWithWeaklySupervisedPreTraining_Path>/data/rest16/train.txt"
    )

    lap14_at_l2 = [
        quad.aspect_term.split(" ")
        for e in lap14
        for quad in e.quads
        if quad.aspect_term != "NULL" and len(quad.aspect_term.split(" ")) == 2
    ]
    rest15_at_l2 = [
        quad.aspect_term.split(" ")
        for e in rest15
        for quad in e.quads
        if quad.aspect_term != "NULL" and len(quad.aspect_term.split(" ")) == 2
    ]
    rest16_at_l2 = [
        quad.aspect_term.split(" ")
        for e in rest16
        for quad in e.quads
        if quad.aspect_term != "NULL" and len(quad.aspect_term.split(" ")) == 2
    ]

    lap14_at_l3 = [
        quad.aspect_term.split(" ")
        for e in lap14
        for quad in e.quads
        if quad.aspect_term != "NULL" and len(quad.aspect_term.split(" ")) == 3
    ]
    rest15_at_l3 = [
        quad.aspect_term.split(" ")
        for e in rest15
        for quad in e.quads
        if quad.aspect_term != "NULL" and len(quad.aspect_term.split(" ")) == 3
    ]
    rest16_at_l3 = [
        quad.aspect_term.split(" ")
        for e in rest16
        for quad in e.quads
        if quad.aspect_term != "NULL" and len(quad.aspect_term.split(" ")) == 3
    ]

    collapse_tags = {
        "NN": "NN*",
        "NNP": "NN*",
        "NNS": "NN*",
        "JJ": "JJ*",
        "JJR": "JJ*",
        "JJS": "JJ*",
        "RB": "RB*",
        "RBR": "RB*",
        "RBS": "RB*",
    }
    # print([nltk.pos_tag(ngram) for ngram in rest15_at_l2])
    # exit()
    lap14_at_l2_tags = [
        [collapse_tags.get(x[1], x[1]) for x in nltk.pos_tag(ngram)]
        for ngram in lap14_at_l2
    ]
    rest15_at_l2_tags = [
        [collapse_tags.get(x[1], x[1]) for x in nltk.pos_tag(ngram)]
        for ngram in rest15_at_l2
    ]
    rest16_at_l2_tags = [
        [collapse_tags.get(x[1], x[1]) for x in nltk.pos_tag(ngram)]
        for ngram in rest16_at_l2
    ]

    lap14_at_l3_tags = [
        [collapse_tags.get(x[1], x[1]) for x in nltk.pos_tag(ngram)]
        for ngram in lap14_at_l3
    ]
    rest15_at_l3_tags = [
        [collapse_tags.get(x[1], x[1]) for x in nltk.pos_tag(ngram)]
        for ngram in rest15_at_l3
    ]
    rest16_at_l3_tags = [
        [collapse_tags.get(x[1], x[1]) for x in nltk.pos_tag(ngram)]
        for ngram in rest16_at_l3
    ]

    df21 = pd.DataFrame(lap14_at_l2_tags, columns=["first word tag", "second word tag"])
    df22 = pd.DataFrame(
        rest15_at_l2_tags, columns=["first word tag", "second word tag"]
    )
    df23 = pd.DataFrame(
        rest16_at_l2_tags, columns=["first word tag", "second word tag"]
    )

    df31 = pd.DataFrame(
        lap14_at_l3_tags,
        columns=["first word tag", "second word tag", "third word tag"],
    )
    df32 = pd.DataFrame(
        rest15_at_l3_tags,
        columns=["first word tag", "second word tag", "third word tag"],
    )
    df33 = pd.DataFrame(
        rest16_at_l3_tags,
        columns=["first word tag", "second word tag", "third word tag"],
    )

    def make_plots_multicolumn(df21, df22, df23, df31, df32, df33):
        fig, ax = plt.subplots(figsize=(9, 6), nrows=1, ncols=2)
        sns.countplot(df21["first word tag"], ax=ax[0], color="darkblue")
        sns.countplot(df21["second word tag"], ax=ax[1], color="darkblue")
        plt.suptitle("SemEval2014 Laptop")
        plt.savefig(
            "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/lap14_bigrams.png"
        )
        plt.cla()
        plt.clf()

        fig, ax = plt.subplots(figsize=(9, 6), nrows=1, ncols=2)
        sns.countplot(df22["first word tag"], ax=ax[0], color="darkblue")
        sns.countplot(df22["second word tag"], ax=ax[1], color="darkblue")
        plt.suptitle("SemEval2015 Restaurant")
        plt.savefig(
            "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/rest15_bigrams.png"
        )
        plt.cla()
        plt.clf()

        fig, ax = plt.subplots(figsize=(9, 6), nrows=1, ncols=2)
        sns.countplot(df23["first word tag"], ax=ax[0], color="darkblue")
        sns.countplot(df23["second word tag"], ax=ax[1], color="darkblue")
        plt.suptitle("SemEval2014 Laptop")
        plt.savefig(
            "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/rest16_bigrams.png"
        )
        plt.cla()
        plt.clf()

        fig, ax = plt.subplots(figsize=(15, 6), nrows=1, ncols=3)
        sns.countplot(df31["first word tag"], ax=ax[0], color="darkblue")
        sns.countplot(df31["second word tag"], ax=ax[1], color="darkblue")
        sns.countplot(df31["third word tag"], ax=ax[2], color="darkblue")
        plt.suptitle("SemEval2014 Laptop")
        plt.savefig(
            "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/lap14_trigrams.png"
        )
        plt.cla()
        plt.clf()

        fig, ax = plt.subplots(figsize=(15, 6), nrows=1, ncols=3)
        sns.countplot(df32["first word tag"], ax=ax[0], color="darkblue")
        sns.countplot(df32["second word tag"], ax=ax[1], color="darkblue")
        sns.countplot(df32["third word tag"], ax=ax[2], color="darkblue")
        plt.suptitle("SemEval2015 Restaurant")
        plt.savefig(
            "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/rest15_trigrams.png"
        )
        plt.cla()
        plt.clf()

        fig, ax = plt.subplots(figsize=(15, 6), nrows=1, ncols=3)
        sns.countplot(df33["first word tag"], ax=ax[0], color="darkblue")
        sns.countplot(df33["second word tag"], ax=ax[1], color="darkblue")
        sns.countplot(df33["third word tag"], ax=ax[2], color="darkblue")
        plt.suptitle("SemEval2014 Laptop")
        plt.savefig(
            "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/rest16_trigrams.png"
        )
        plt.cla()
        plt.clf()

    def make_plots_merged(
        lap14_at_l2_tags,
        rest15_at_l2_tags,
        rest16_at_l2_tags,
        lap14_at_l3_tags,
        rest15_at_l3_tags,
        rest16_at_l3_tags,
    ):
        lap14_at_l2_tags_merged = ["-".join(x) for x in lap14_at_l2_tags]
        rest15_at_l2_tags_merged = ["-".join(x) for x in rest15_at_l2_tags]
        rest16_at_l2_tags_merged = ["-".join(x) for x in rest16_at_l2_tags]
        lap14_at_l3_tags_merged = ["-".join(x) for x in lap14_at_l3_tags]
        rest15_at_l3_tags_merged = ["-".join(x) for x in rest15_at_l3_tags]
        rest16_at_l3_tags_merged = ["-".join(x) for x in rest16_at_l3_tags]

        fig, ax = plt.subplots(figsize=(14, 12), nrows=1, ncols=1)
        ax = sns.countplot(lap14_at_l2_tags_merged, ax=ax, color="darkblue")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40)  # , ha="right")
        plt.xlabel("POS Tag Pattern", fontsize=16)
        plt.ylabel("Count", fontsize=16)
        plt.title("SemEval2014 Laptop Bigrams", fontsize=16)
        plt.savefig(
            "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/lap14_bigrams_merged.png"
        )
        plt.clf()
        plt.cla()

        fig, ax = plt.subplots(figsize=(14, 12), nrows=1, ncols=1)
        ax = sns.countplot(lap14_at_l3_tags_merged, ax=ax, color="darkblue")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40)  # , ha="right")
        plt.xlabel("POS Tag Pattern", fontsize=16)
        plt.ylabel("Count", fontsize=16)
        plt.title("SemEval2014 Laptop Trigrams", fontsize=16)
        plt.savefig(
            "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/lap14_trigrams_merged.png"
        )
        plt.clf()
        plt.cla()

        fig, ax = plt.subplots(figsize=(14, 12), nrows=1, ncols=1)
        ax = sns.countplot(rest15_at_l2_tags_merged, ax=ax, color="darkblue")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40)  # , ha="right")
        plt.xlabel("POS Tag Pattern", fontsize=16)
        plt.ylabel("Count", fontsize=16)
        plt.title("SemEval2015 Restaurant", fontsize=16)
        plt.savefig(
            "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/rest15_bigrams_merged.png"
        )
        plt.clf()
        plt.cla()

        fig, ax = plt.subplots(figsize=(14, 12), nrows=1, ncols=1)
        ax = sns.countplot(rest15_at_l3_tags_merged, ax=ax, color="darkblue")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40)  # , ha="right")
        plt.xlabel("POS Tag Pattern", fontsize=16)
        plt.ylabel("Count", fontsize=16)
        plt.title("SemEval2015 Restaurant", fontsize=16)
        plt.savefig(
            "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/rest15_trigrams_merged.png"
        )
        plt.clf()
        plt.cla()

        fig, ax = plt.subplots(figsize=(14, 12), nrows=1, ncols=1)
        ax = sns.countplot(rest16_at_l2_tags_merged, ax=ax, color="darkblue")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40)  # , ha="right")
        plt.xlabel("POS Tag Pattern", fontsize=16)
        plt.ylabel("Count", fontsize=16)
        plt.title("SemEval2016 Restaurant", fontsize=16)
        plt.savefig(
            "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/rest16_bigrams_merged.png"
        )
        plt.clf()
        plt.cla()

        fig, ax = plt.subplots(figsize=(14, 12), nrows=1, ncols=1)
        ax = sns.countplot(rest16_at_l3_tags_merged, ax=ax, color="darkblue")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40)  # , ha="right")
        plt.xlabel("POS Tag Pattern", fontsize=16)
        plt.ylabel("Count", fontsize=16)
        plt.title("SemEval2016 Restaurant", fontsize=16)
        plt.savefig(
            "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/rest16_trigrams_merged.png"
        )
        plt.clf()
        plt.cla()

    make_plots_multicolumn(df21, df22, df23, df31, df32, df33)
    make_plots_merged(
        lap14_at_l2_tags,
        rest15_at_l2_tags,
        rest16_at_l2_tags,
        lap14_at_l3_tags,
        rest15_at_l3_tags,
        rest16_at_l3_tags,
    )


def analyze_metrics(
    unannotated_dataset,
    gold_dataset: List[ABSAExample],
    freq_filter: Dict[str, int] = {},
):
    gold_dataset_at_l2 = [
        tuple(quad.aspect_term.split(" "))
        for e in gold_dataset
        for quad in e.quads
        if quad.aspect_term != "NULL" and len(quad.aspect_term.split(" ")) == 2
    ]
    gold_dataset_at_l3 = [
        tuple(quad.aspect_term.split(" "))
        for e in gold_dataset
        for quad in e.quads
        if quad.aspect_term != "NULL" and len(quad.aspect_term.split(" ")) == 3
    ]

    collapsed = [y for x in unannotated_dataset["cleaned_review_body"] for y in x]

    bigrams = nltk.collocations.BigramAssocMeasures()
    trigrams = nltk.collocations.TrigramAssocMeasures()
    quadgrams = nltk.collocations.QuadgramAssocMeasures()

    bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(collapsed)
    print("Bigrams computed")
    trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(collapsed)
    print("Trigrams computed")
    quadgramFinder = nltk.collocations.QuadgramCollocationFinder.from_words(collapsed)
    print("Quadgrams computed")

    bigramFinder.apply_freq_filter(freq_filter.get("bigram", 20))
    trigramFinder.apply_freq_filter(freq_filter.get("trigram", 20))
    quadgramFinder.apply_freq_filter(freq_filter.get("quadgram", 20))

    bigrams_df = pd.DataFrame(
        list(
            zip(
                [
                    x[0]
                    for x in sorted(
                        bigramFinder.score_ngrams(bigrams.pmi), key=lambda x: x[0]
                    )
                ],
                [
                    x[1]
                    for x in sorted(bigramFinder.ngram_fd.items(), key=lambda x: x[0])
                ],
                [
                    x[1]
                    for x in sorted(
                        bigramFinder.score_ngrams(bigrams.pmi), key=lambda x: x[0]
                    )
                ],
                [
                    x[1]
                    for x in sorted(
                        bigramFinder.score_ngrams(bigrams.student_t), key=lambda x: x[0]
                    )
                ],
                [
                    x[1]
                    for x in sorted(
                        bigramFinder.score_ngrams(bigrams.chi_sq), key=lambda x: x[0]
                    )
                ],
                [
                    x[1]
                    for x in sorted(
                        bigramFinder.score_ngrams(bigrams.likelihood_ratio),
                        key=lambda x: x[0],
                    )
                ],
            )
        ),
        columns=["bigram", "count", "PMI", "t", "chi-sq", "Likelihood"],
    ).sort_values(by="PMI", ascending=False)

    trigrams_df = pd.DataFrame(
        list(
            zip(
                [
                    x[0]
                    for x in sorted(
                        trigramFinder.score_ngrams(trigrams.pmi), key=lambda x: x[0]
                    )
                ],
                [
                    x[1]
                    for x in sorted(trigramFinder.ngram_fd.items(), key=lambda x: x[0])
                ],
                [
                    x[1]
                    for x in sorted(
                        trigramFinder.score_ngrams(trigrams.pmi), key=lambda x: x[0]
                    )
                ],
                [
                    x[1]
                    for x in sorted(
                        trigramFinder.score_ngrams(trigrams.student_t),
                        key=lambda x: x[0],
                    )
                ],
                [
                    x[1]
                    for x in sorted(
                        trigramFinder.score_ngrams(trigrams.chi_sq), key=lambda x: x[0]
                    )
                ],
                [
                    x[1]
                    for x in sorted(
                        trigramFinder.score_ngrams(trigrams.likelihood_ratio),
                        key=lambda x: x[0],
                    )
                ],
            )
        ),
        columns=["trigram", "count", "PMI", "t", "chi-sq", "Likelihood"],
    ).sort_values(by="PMI", ascending=False)

    quadgrams_df = pd.DataFrame(
        list(
            zip(
                [
                    x[0]
                    for x in sorted(
                        quadgramFinder.score_ngrams(quadgrams.pmi), key=lambda x: x[0]
                    )
                ],
                [
                    x[1]
                    for x in sorted(quadgramFinder.ngram_fd.items(), key=lambda x: x[0])
                ],
                [
                    x[1]
                    for x in sorted(
                        quadgramFinder.score_ngrams(quadgrams.pmi), key=lambda x: x[0]
                    )
                ],
                [
                    x[1]
                    for x in sorted(
                        quadgramFinder.score_ngrams(quadgrams.student_t),
                        key=lambda x: x[0],
                    )
                ],
                [
                    x[1]
                    for x in sorted(
                        quadgramFinder.score_ngrams(quadgrams.chi_sq),
                        key=lambda x: x[0],
                    )
                ],
                [
                    x[1]
                    for x in sorted(
                        quadgramFinder.score_ngrams(quadgrams.likelihood_ratio),
                        key=lambda x: x[0],
                    )
                ],
            )
        ),
        columns=["quadgram", "count", "PMI", "t", "chi-sq", "Likelihood"],
    ).sort_values(by="PMI", ascending=False)

    gold_bigrams_df = bigrams_df[
        bigrams_df.apply(lambda x: x["bigram"] in gold_dataset_at_l2, axis=1)
    ]
    gold_trigrams_df = trigrams_df[
        trigrams_df.apply(lambda x: x["trigram"] in gold_dataset_at_l3, axis=1)
    ]

    return (bigrams_df, trigrams_df, quadgrams_df, gold_bigrams_df, gold_trigrams_df)


def plot_analyze_metrics_rest1516():
    rest15 = read_semeval_format(
        "<ABSAWithWeaklySupervisedPreTraining_Path>/data/rest15/train.txt"
    )
    rest16 = read_semeval_format(
        "<ABSAWithWeaklySupervisedPreTraining_Path>/data/rest16/train.txt"
    )

    # train = datasets.load_dataset('amazon_us_reviews', 'Electronics_v1_00')['train'].train_test_split(test_size=0.2, seed=1)['train'].map(lambda x: {'cleaned_review_body': map_clean_comments(x['review_body'])}, batched=True, batch_size=5000)
    # train.save_to_disk('<ABSAWithWeaklySupervisedPreTraining_Path>/cache/amazon_us_reviews_electronicsv100_commit=944b6ef7de4bf6ff594b28ea98c606b820acb5ed')
    # train = datasets.load_dataset('yelp_review_full')['train'].train_test_split(test_size=0.1, seed=1)['train'].map(lambda x: {'cleaned_review_body': map_clean_comments(x['text'])}, batched=True, batch_size=2500)
    # train.save_to_disk('<ABSAWithWeaklySupervisedPreTraining_Path>/cache/yelp_review_full_commit=944b6ef7de4bf6ff594b28ea98c606b820acb5ed')

    train = datasets.load_from_disk(
        "<ABSAWithWeaklySupervisedPreTraining_Path>/cache/yelp_review_full_commit=944b6ef7de4bf6ff594b28ea98c606b820acb5ed"
    )
    print("Dataset loaded from disk")
    (
        bigrams_df_rest,
        trigrams_df_rest,
        quadgrams_df_rest,
        gold_bigrams_df_rest,
        gold_trigrams_df_rest,
    ) = analyze_metrics(train, rest15 + rest16)
    bigrams_df, trigrams_df, _ = build_ngrams(
        datasets.load_from_disk(
            "<ABSAWithWeaklySupervisedPreTraining_Path>/cache/yelp_review_full_commit=944b6ef7de4bf6ff594b28ea98c606b820acb5ed"
        ),
        compute_ngrams={"bigrams": True, "trigrams": True},
    )

    print("Metrics computed")

    sns.scatterplot(
        data=bigrams_df,
        x="PMI",
        y="Likelihood",
        color="black",
        marker="x",
        label="Yelp Reviews",
    )
    sns.scatterplot(
        data=gold_bigrams_df_rest,
        x="PMI",
        y="Likelihood",
        color="blue",
        marker="*",
        s=80,
        label="SemEval2015-2016 Restaurant",
    )
    plt.yscale("log")
    plt.xlabel("PMI")
    plt.ylabel("Likelihood (log)")
    plt.title(
        "Comparing metrics on bigram aspect terms\nfrom SemEval2015-2016 Restaurant with Yelp Reviews"
    )
    plt.savefig(
        "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/rest1516_ate_comparison_pmi_likelihood.png"
    )
    plt.clf()
    plt.cla()

    sns.scatterplot(
        data=bigrams_df,
        x="PMI",
        y="count",
        color="black",
        marker="x",
        label="Yelp Reviews",
    )
    sns.scatterplot(
        data=gold_bigrams_df_rest,
        x="PMI",
        y="count",
        color="blue",
        marker="*",
        s=80,
        label="SemEval2015-2016 Restaurant",
    )
    plt.yscale("log")
    plt.xlabel("PMI")
    plt.ylabel("count (log)")
    plt.title(
        "Comparing metrics on bigram aspect terms\nfrom SemEval2015-2016 Restaurant with Yelp Reviews"
    )
    plt.savefig(
        "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/rest1516_ate_comparison_pmi_count.png"
    )
    plt.clf()
    plt.cla()

    sns.scatterplot(
        data=bigrams_df,
        x="PMI",
        y="chi-sq",
        color="black",
        marker="x",
        label="Yelp Reviews",
    )
    sns.scatterplot(
        data=gold_bigrams_df_rest,
        x="PMI",
        y="chi-sq",
        color="blue",
        marker="*",
        s=80,
        label="SemEval2015-2016 Restaurant",
    )
    plt.yscale("log")
    plt.xlabel("PMI")
    plt.ylabel("chi-sq (log)")
    plt.title(
        "Comparing metrics on bigram aspect terms\nfrom SemEval2015-2016 Restaurant with Yelp Reviews"
    )
    plt.savefig(
        "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/rest1516_ate_comparison_pmi_chisq.png"
    )
    plt.clf()
    plt.cla()

    sns.scatterplot(
        data=bigrams_df,
        x="Likelihood",
        y="chi-sq",
        color="black",
        marker="x",
        label="Yelp Reviews",
    )
    sns.scatterplot(
        data=gold_bigrams_df_rest,
        x="Likelihood",
        y="chi-sq",
        color="blue",
        marker="*",
        s=80,
        label="SemEval2015-2016 Restaurant",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Likelihood (log)")
    plt.ylabel("chi-sq (log)")
    plt.title(
        "Comparing metrics on bigram aspect terms\nfrom SemEval2016 Restaurant with Yelp Reviews"
    )
    plt.savefig(
        "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/rest1516_ate_comparison_likelihood_chisq.png"
    )
    plt.clf()
    plt.cla()

    # The return is to enable playing with the variables if we run it with -i flag
    return (
        bigrams_df_rest,
        trigrams_df_rest,
        quadgrams_df_rest,
        gold_bigrams_df_rest,
        gold_trigrams_df_rest,
    )


def plot_analyze_metrics_lap14():
    lap14 = read_semeval_format(
        "<ABSAWithWeaklySupervisedPreTraining_Path>/data/lap14/train.txt"
    )

    # train = datasets.load_dataset('amazon_us_reviews', 'Electronics_v1_00')['train'].train_test_split(test_size=0.2, seed=1)['train'].map(lambda x: {'cleaned_review_body': map_clean_comments(x['review_body'])}, batched=True, batch_size=5000)
    # train.save_to_disk('<ABSAWithWeaklySupervisedPreTraining_Path>/cache/amazon_us_reviews_electronicsv100_commit=944b6ef7de4bf6ff594b28ea98c606b820acb5ed')
    # train = datasets.load_dataset('yelp_review_full')['train'].train_test_split(test_size=0.1, seed=1)['train'].map(lambda x: {'cleaned_review_body': map_clean_comments(x['text'])}, batched=True, batch_size=2500)
    # train.save_to_disk('<ABSAWithWeaklySupervisedPreTraining_Path>/cache/yelp_review_full_commit=944b6ef7de4bf6ff594b28ea98c606b820acb5ed')

    train = datasets.load_from_disk(
        "<ABSAWithWeaklySupervisedPreTraining_Path>/cache/amazon_us_reviews_electronicsv100_commit=944b6ef7de4bf6ff594b28ea98c606b820acb5ed"
    )
    print("Dataset loaded from disk")
    (
        bigrams_df_lap,
        trigrams_df_lap,
        quadgrams_df_lap,
        gold_bigrams_df_lap,
        gold_trigrams_df_lap,
    ) = analyze_metrics(train, lap14)
    print("Metrics computed")

    sns.scatterplot(
        data=bigrams_df_lap,
        x="PMI",
        y="Likelihood",
        color="black",
        marker="x",
        label="Amazon Electronics",
    )
    sns.scatterplot(
        data=gold_bigrams_df_lap,
        x="PMI",
        y="Likelihood",
        color="blue",
        marker="*",
        s=80,
        label="SemEval2014 Laptop",
    )
    plt.yscale("log")
    plt.xlabel("PMI")
    plt.ylabel("Likelihood (log)")
    plt.title(
        "Comparing metrics on bigram aspect terms\nfrom SemEval2014 Laptop with Amazon Electronics"
    )
    plt.savefig(
        "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/metrics/lap14/lap14_ate_comparison_pmi_likelihood.png"
    )
    plt.clf()
    plt.cla()

    sns.scatterplot(
        data=bigrams_df_lap,
        x="PMI",
        y="count",
        color="black",
        marker="x",
        label="Amazon Electronics",
    )
    sns.scatterplot(
        data=gold_bigrams_df_lap,
        x="PMI",
        y="count",
        color="blue",
        marker="*",
        s=80,
        label="SemEval2014 Laptop",
    )
    plt.yscale("log")
    plt.xlabel("PMI")
    plt.ylabel("count (log)")
    plt.title(
        "Comparing metrics on bigram aspect terms\nfrom SemEval2014 Laptop with Amazon Electronics"
    )
    plt.savefig(
        "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/metrics/lap14/lap14_ate_comparison_pmi_count.png"
    )
    plt.clf()
    plt.cla()

    sns.scatterplot(
        data=bigrams_df_lap,
        x="PMI",
        y="chi-sq",
        color="black",
        marker="x",
        label="Amazon Electronics",
    )
    sns.scatterplot(
        data=gold_bigrams_df_lap,
        x="PMI",
        y="chi-sq",
        color="blue",
        marker="*",
        s=80,
        label="SemEval2014 Laptop",
    )
    plt.yscale("log")
    plt.xlabel("PMI")
    plt.ylabel("chi-sq (log)")
    plt.title(
        "Comparing metrics on bigram aspect terms\nfrom SemEval2014 Laptop with Amazon Electronics"
    )
    plt.savefig(
        "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/metrics/lap14/lap14_ate_comparison_pmi_chisq.png"
    )
    plt.clf()
    plt.cla()

    sns.scatterplot(
        data=bigrams_df_lap,
        x="Likelihood",
        y="chi-sq",
        color="black",
        marker="x",
        label="Amazon Electronics",
    )
    sns.scatterplot(
        data=gold_bigrams_df_lap,
        x="Likelihood",
        y="chi-sq",
        color="blue",
        marker="*",
        s=80,
        label="SemEval2014 Laptop",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Likelihood (log)")
    plt.ylabel("chi-sq (log)")
    plt.title(
        "Comparing metrics on bigram aspect terms\nfrom SemEval2014 Laptop with Amazon Electronics"
    )
    plt.savefig(
        "<ABSAWithWeaklySupervisedPreTraining_Path>/results/week5/ate_analyze/metrics/lap14/lap14_ate_comparison_likelihood_chisq.png"
    )
    plt.clf()
    plt.cla()

    # The return is to enable playing with the variables if we run it with -i flag
    return (
        bigrams_df_lap,
        trigrams_df_lap,
        quadgrams_df_lap,
        gold_bigrams_df_lap,
        gold_trigrams_df_lap,
    )


if __name__ == "__main__":
    (bigrams_df, trigrams_df, quadgrams_df, gold_bigrams_df, gold_trigrams_df) = (
        plot_analyze_metrics_rest1516()
    )
    # (bigrams_df, trigrams_df, quadgrams_df, gold_bigrams_df, gold_trigrams_df) = plot_analyze_metrics_lap14()
