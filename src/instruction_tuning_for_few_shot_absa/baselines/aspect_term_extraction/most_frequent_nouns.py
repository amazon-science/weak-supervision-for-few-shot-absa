"""
Extract aspect terms by looking at the most frequent words per domain
Motivation: Embarrassingly Simple Unsupervised Aspect Extraction (https://aclanthology.org/2020.acl-main.290.pdf)

Note: Used https://medium.com/@nicharuch/collocations-identifying-phrases-that-act-like-individual-words-in-nlp-f58a93a2f84a as inspiration
"""

import argparse
import json
from typing import Dict, List, Tuple
from collections import Counter
from spacy.matcher import Matcher
import nltk
import tqdm
import spacy
import multiprocessing
import timeit
import datasets
import pandas as pd


nlp = spacy.load("en_core_web_sm")

with open("data/lexicon/merged/lexicon.txt") as fin:
    sentiment_words = [x.strip() for x in fin.readlines()]

en_stopwords = set(nltk.corpus.stopwords.words("english")).union(
    ["http", "https", "www", "com"]
)
en_stopwords_and_sentiment_words = en_stopwords.union(sentiment_words)

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
acceptable_types = set(
    [
        "NN*-NN*",
        "JJ*-NN*",
        "VBG-NN*",
        "VBN-NN*",
        "NN*-NN*-NN*",
        "NN*-IN-NN*",
        "JJ*-NN*-NN*",
        "JJ*-JJ*-NN*",
        "VBN-JJ*-NN*",
        "NN*-NN*-NN*-NN*",
        "NN*-CC-NN*-NN*",
        "VBN-NN*-NCC-NN*",
    ]
)


"""
Get a Counter from the sentences
Assumes the sentences are already cleaned
"""


def get_noun_count_dict(sentences: List[str]) -> Counter:
    nouns = []
    for s in tqdm.tqdm(sentences):
        doc = nlp(s)
        n = [
            x.lemma_.lower()
            for x in doc
            if "NN" in x.tag_
            and ">" not in x.text
            and "<" not in x.text
            and x.text != "br"
        ]
        nouns.append(n)

    return Counter([y for x in nouns for y in x])


def extract_from_dataset(dataset):
    sharded = [
        dataset.shard(num_shards=8, index=0)["review_body"],
        dataset.shard(num_shards=8, index=1)["review_body"],
        dataset.shard(num_shards=8, index=2)["review_body"],
        dataset.shard(num_shards=8, index=3)["review_body"],
        dataset.shard(num_shards=8, index=4)["review_body"],
        dataset.shard(num_shards=8, index=5)["review_body"],
        dataset.shard(num_shards=7, index=6)["review_body"],
        dataset.shard(num_shards=8, index=7)["review_body"],
    ]
    pool = multiprocessing.Pool(processes=8)
    start_time = timeit.default_timer()
    result = pool.map(get_noun_count_dict, sharded)
    final_result_nouns = Counter()
    final_result_noun_chunks = Counter()
    for r1, r2 in result:
        final_result_nouns = final_result_nouns + r1
        final_result_noun_chunks = final_result_noun_chunks + r2
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    print("---------")
    result1 = {
        k: v for (k, v) in sorted(final_result_nouns.items(), key=lambda x: -x[1])
    }
    result2 = {
        k: v for (k, v) in sorted(final_result_noun_chunks.items(), key=lambda x: -x[1])
    }
    return (result1, result2)


# function to filter for ADJ/NN bigrams
def ngram_filter(ngram):
    # if '-pron-' in ngram or '' in ngram or ' 'in ngram or 't' in ngram:
    #     return False
    for word in ngram:
        if word in en_stopwords_and_sentiment_words:
            return False
    tags = nltk.pos_tag(ngram)
    # Collapse the tags (e.g. NNP -> NN*, NNS -> NN*, etc), then merge them with '-'
    # Then check if it is in the set
    tags = [collapse_tags.get(x[1], x[1]) for x in tags]
    if "-".join(tags) in acceptable_types or (
        len(tags) > 3 and len([x for x in tags if "NN" in x]) > 1
    ):
        return True
    else:
        return False


def build_ngrams(
    unannotated_dataset,
    compute_ngrams: Dict[str, bool],
    freq_filter: Dict[str, int] = {},
):
    # unannotated_dataset = datasets.load_from_disk('<ABSAWithWeaklySupervisedPreTraining_Path>n/cache/yelp_review_full_commit=944b6ef7de4bf6ff594b28ea98c606b820acb5ed')
    collapsed = [y for x in unannotated_dataset["cleaned_review_body"] for y in x]

    bigrams_df, trigrams_df, quadgrams_df = None, None, None
    if compute_ngrams.get("bigrams", False):
        bigrams = nltk.collocations.BigramAssocMeasures()
        bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(collapsed)
        bigramFinder.apply_freq_filter(freq_filter.get("bigram", 20))
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
                        for x in sorted(
                            bigramFinder.ngram_fd.items(), key=lambda x: x[0]
                        )
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
                            bigramFinder.score_ngrams(bigrams.student_t),
                            key=lambda x: x[0],
                        )
                    ],
                    [
                        x[1]
                        for x in sorted(
                            bigramFinder.score_ngrams(bigrams.chi_sq),
                            key=lambda x: x[0],
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
            columns=["ngram", "count", "PMI", "t", "chi-sq", "Likelihood"],
        ).sort_values(by="PMI", ascending=False)
        print("Bigrams computed")
    if compute_ngrams.get("trigrams", False):
        trigrams = nltk.collocations.TrigramAssocMeasures()
        trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(collapsed)
        trigramFinder.apply_freq_filter(freq_filter.get("trigram", 20))
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
                        for x in sorted(
                            trigramFinder.ngram_fd.items(), key=lambda x: x[0]
                        )
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
                            trigramFinder.score_ngrams(trigrams.chi_sq),
                            key=lambda x: x[0],
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
            columns=["ngram", "count", "PMI", "t", "chi-sq", "Likelihood"],
        ).sort_values(by="PMI", ascending=False)
        print("Trigrams computed")
    if compute_ngrams.get("quadgrams", False):
        quadgrams = nltk.collocations.QuadgramAssocMeasures()
        quadgramFinder = nltk.collocations.QuadgramCollocationFinder.from_words(
            collapsed
        )
        quadgramFinder.apply_freq_filter(freq_filter.get("quadgram", 20))
        quadgrams_df = pd.DataFrame(
            list(
                zip(
                    [
                        x[0]
                        for x in sorted(
                            quadgramFinder.score_ngrams(quadgrams.pmi),
                            key=lambda x: x[0],
                        )
                    ],
                    [
                        x[1]
                        for x in sorted(
                            quadgramFinder.ngram_fd.items(), key=lambda x: x[0]
                        )
                    ],
                    [
                        x[1]
                        for x in sorted(
                            quadgramFinder.score_ngrams(quadgrams.pmi),
                            key=lambda x: x[0],
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
            columns=["ngram", "count", "PMI", "t", "chi-sq", "Likelihood"],
        ).sort_values(by="PMI", ascending=False)
        print("Quadgrams computed")

    # TODO Counter(collapsed)
    return bigrams_df, trigrams_df, quadgrams_df


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        default="",
        type=str,
        required=True,
        help="Name of the dataset (e.g. `amazon_us_reviews`)",
    )
    parser.add_argument(
        "--subset_name",
        default=None,
        type=str,
        required=False,
        help="Name of the subset (e.g. `Books_v1_00`)",
    )
    parser.add_argument(
        "--save_path",
        default="",
        type=str,
        required=True,
        help="Where to save the dictionary",
    )
    parser.add_argument(
        "--train_test_split",
        action="store_true",
        help="If set, it will split into train and test and run the counter only on train",
    )
    parser.add_argument(
        "--train_test_split_size",
        default=None,
        required=False,
        type=float,
        help="The size of test, in case the --train_test_split flag is set",
    )
    parser.add_argument(
        "--seed",
        default=None,
        required=False,
        type=int,
        help="The seed to set to the `train_test_split` call, in case the --train_test_split flag is set",
    )

    return parser


def extract_ate_from_sentence(
    sentence: str, matchers: List[Matcher]
) -> Tuple[List[str], List[str]]:
    doc = nlp(sentence)
    ates_present = []
    indices_in = set()
    for matcher in matchers:
        for _, start, end in matcher(doc):
            if start not in indices_in:
                ates_present.append((" ".join([x.text for x in doc[start:end]]), start))
                for i in range(start, end):
                    indices_in.add(i)
    ates_present = [x[0] for x in sorted(ates_present, key=lambda k: k[1])]
    if len(ates_present) > 0:
        return ([x.text for x in doc], ates_present)
    else:
        return ([x.text for x in doc], [])


def extract_ate_from_multiple_sentences(
    sentences: List[str], matchers: List[Matcher]
) -> List[Tuple[List[str], List[str]]]:
    return [extract_ate_from_sentence(sentence, matchers) for sentence in sentences]


# Getting around two argument functions and how lambda functions are not pickable (when using multiprocessing.Pool)
def do_matching_on_subset(shard_matchers):
    (shard, matchers) = shard_matchers
    result = []
    for sentence in tqdm.tqdm(shard):
        result.append(extract_ate_from_sentence(sentence, matchers))
    return result


def get_candidate_ates(
    bigrams_df, trigrams_df, quadgrams_df, collapsed, take_top_n_nouns=5000
):
    bigrams_df_filtered = bigrams_df[bigrams_df.ngram.map(lambda x: ngram_filter(x))]
    bigrams_df_filtered = bigrams_df_filtered[bigrams_df_filtered["PMI"] > 0]

    trigrams_df_filtered = trigrams_df[trigrams_df.ngram.map(lambda x: ngram_filter(x))]
    trigrams_df_filtered = trigrams_df_filtered[
        trigrams_df_filtered.ngram.map(lambda x: len(set(x)) > 1)
    ]
    trigrams_df_filtered = trigrams_df_filtered[trigrams_df_filtered["PMI"] > 0]

    quadgrams_df_filtered = quadgrams_df[
        quadgrams_df.ngram.map(lambda x: ngram_filter(x))
    ]
    quadgrams_df_filtered = quadgrams_df_filtered[
        quadgrams_df_filtered.ngram.map(lambda x: len(set(x)) > 1)
    ]
    quadgrams_df_filtered = quadgrams_df_filtered[quadgrams_df_filtered["PMI"] > 0]

    collapsed_counter = Counter(collapsed)
    print(f"Total number of nouns: {len(collapsed_counter)}")
    nouns_collapsed_counter = []
    for word, count in tqdm.tqdm(
        list(sorted(collapsed_counter.items(), key=lambda x: -x[1]))
    ):
        if (
            len(nouns_collapsed_counter) < take_top_n_nouns
            and word not in en_stopwords_and_sentiment_words
            and "NN" in nlp(word)[0].tag_
        ):
            nouns_collapsed_counter.append((word, count))

    return (
        nouns_collapsed_counter,
        bigrams_df_filtered,
        trigrams_df_filtered,
        quadgrams_df_filtered,
    )


"""
:param data                   -> on what data to apply this function; should be a huggingface dataset type of data (so list of dictionaries)
:param saving_path            -> where to save the output
:param freq_filter            -> frequency filter for bigram, trigram and quad gram
:param split_long_sentences   -> boolean flag for whether to split sentences that are very long
:param skip_empty_extractions -> boolean flag for whether sentences where nothing was extracted should be included in the output or not
:param text_column_name       -> the name of the text column for :param data (since we use a huggingface dataset type of data)
                                 Importantly, this is not the column appended with the cleaned text, but the original columns and we use it to split
                                 the lines into multiple; We differentiate like this because, at the end of the day, we are interested in the original
                                 review, but for computing counts we want a clean version
:param take_top_n_nouns       -> how many nouns to take for unigram
"""


def extract_ate(
    data,
    saving_path="logs/baselines/week6/ate_dataset/yelp_with_ates",
    freq_filter={},
    split_long_sentences=True,
    skip_empty_extractions=True,
    text_column_name="raw_text",
    take_top_n_nouns=5000,
):

    bigrams_df, trigrams_df, quadgrams_df = build_ngrams(
        data["train"],
        compute_ngrams={"bigrams": True, "trigrams": True, "quadgrams": True},
        freq_filter=freq_filter,
    )

    collapsed = [y for x in data["train"]["cleaned_review_body"] for y in x]

    (
        nouns_collapsed_counter,
        bigrams_df_filtered,
        trigrams_df_filtered,
        quadgrams_df_filtered,
    ) = get_candidate_ates(
        bigrams_df, trigrams_df, quadgrams_df, collapsed, take_top_n_nouns
    )

    matchers = []

    for ngrams in [
        quadgrams_df_filtered.ngram.tolist(),
        trigrams_df_filtered.ngram.tolist(),
        bigrams_df_filtered.ngram.tolist(),
    ]:
        matcher = Matcher(nlp.vocab)
        for ate in ngrams:
            matcher.add("ate", [[{"LEMMA": x} for x in ate]])
        matchers.append(matcher)

    matcher_sw = Matcher(nlp.vocab)
    for ate in [n[0].strip() for n in nouns_collapsed_counter[:take_top_n_nouns]]:
        matcher_sw.add("ate", [[{"LEMMA": ate}]])
    matchers.append(matcher_sw)

    input_data = []
    for line in tqdm.tqdm(data["train"][text_column_name]):
        sent_line = nltk.sent_tokenize(line)
        if split_long_sentences and len(sent_line) > 3:
            for s in sent_line:
                input_data.append(s)
        else:
            input_data.append(line)

    step_size = len(input_data) // 8 + 1
    pool = multiprocessing.Pool(8)
    sharded = [
        input_data[(0 * step_size) : (1 * step_size)],
        input_data[(1 * step_size) : (2 * step_size)],
        input_data[(2 * step_size) : (3 * step_size)],
        input_data[(3 * step_size) : (4 * step_size)],
        input_data[(4 * step_size) : (5 * step_size)],
        input_data[(5 * step_size) : (6 * step_size)],
        input_data[(6 * step_size) : (7 * step_size)],
        input_data[(7 * step_size) :],
    ]
    sharded = [(x, matchers) for x in sharded]
    pool_result = pool.map(do_matching_on_subset, sharded)

    result = [
        {"sentence": " ".join(y[0]).strip(), "quads": [[z, "", "", ""] for z in y[1]]}
        for x in pool_result
        for y in x
    ]

    if skip_empty_extractions:
        result = [x for x in result if len(x["quads"]) > 0]

    if saving_path:
        with open(saving_path, "w+") as fout:
            for line in tqdm.tqdm(result):
                fout.write(f"{json.dumps(line)}\n")

    return


if __name__ == "__main__":
    # data = datasets.load_dataset('json', data_files='data/yelp_expanded_subset_better/yelp_academic_dataset_review_uf_6250.json').map(lambda x: {'cleaned_review_body': map_clean_comments(x['review_body'])}, batched=True, batch_size=5000)
    data = datasets.load_from_disk(
        "cache/yelp_academic_dataset_review_uf_6250_944b6ef7"
    )
    extract_ate(data)
    # bigrams_df, trigrams_df, quadgrams_df = build_ngrams(
    #     data['train'],
    #     compute_ngrams={'bigrams': True, 'trigrams': True, 'quadgrams': True}
    # )

    # bigrams_df_filtered   = bigrams_df[bigrams_df.ngram.map(lambda x: ngram_filter(x))]
    # bigrams_df_filtered   = bigrams_df_filtered[bigrams_df_filtered['PMI'] > 0]

    # trigrams_df_filtered  = trigrams_df[trigrams_df.ngram.map(lambda x: ngram_filter(x))]
    # trigrams_df_filtered  = trigrams_df_filtered[trigrams_df_filtered.ngram.map(lambda x: len(set(x)) > 1)]
    # trigrams_df_filtered  = trigrams_df_filtered[trigrams_df_filtered['PMI'] > 0]

    # quadgrams_df_filtered = quadgrams_df[quadgrams_df.ngram.map(lambda x: ngram_filter(x))]
    # quadgrams_df_filtered = quadgrams_df_filtered[quadgrams_df_filtered.ngram.map(lambda x: len(set(x)) > 1)]
    # quadgrams_df_filtered = quadgrams_df_filtered[quadgrams_df_filtered['PMI'] > 0]

    # collapsed = [y for x in data['train']['cleaned_review_body'] for y in x]
    # collapsed_counter = Counter(collapsed)
    # nouns_collapsed_counter = []
    # for (word, count) in tqdm.tqdm(list(sorted(collapsed_counter.items(), key=lambda x: -x[1]))):
    #     if len(nouns_collapsed_counter) < 5000 and word not in en_stopwords_and_sentiment_words and ('NN' in nlp(word)[0].tag_ or 'NN' in nltk.pos_tag([word])[0][1]):
    #         nouns_collapsed_counter.append((word, count))

    # matchers = []

    # for ngrams in [
    #     quadgrams_df_filtered.ngram.tolist(),
    #     trigrams_df_filtered.ngram.tolist(),
    #     bigrams_df_filtered.ngram.tolist(),
    # ]:
    #     matcher = Matcher(nlp.vocab)
    #     for ate in ngrams:
    #         matcher.add('ate', [[{'LEMMA': x} for x in ate]])
    #     matchers.append(matcher)

    # matcher_sw = Matcher(nlp.vocab)
    # for ate in [n[0].strip() for n in nouns_collapsed_counter[:5000]]:
    #     matcher_sw.add('ate', [[{'LEMMA': ate}]])
    # matchers.append(matcher_sw)
