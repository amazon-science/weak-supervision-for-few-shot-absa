"""
The idea for these functions is to use the gold aspect terms from train and dev from the SemEval data
to augment our newly constructed dataset.
The motivation is that the performance of the model after being pre-finetuned on such a dataset
should be very high, as it has seen everything there is to see
"""

from itertools import groupby
import datasets
import nltk
import spacy
from spacy.matcher import Matcher
import tqdm
import multiprocessing
from src.weak_supervision_for_few_shot_absa.baselines.aspect_term_extraction.most_frequent_nouns import (
    build_ngrams,
    do_matching_on_subset,
    get_candidate_ates,
)

from src.weak_supervision_for_few_shot_absa.baselines.utils import (
    map_clean_comments,
)

nlp = spacy.load("en_core_web_sm")


def extract_ate(
    data,
    saving_path="logs/baselines/week7/ate_dataset/yelp_augmented_with_gold_withdev.txt",
    freq_filter={},
    split_long_sentences=True,
    skip_empty_extractions=True,
):

    bigrams_df, trigrams_df, quadgrams_df = build_ngrams(
        data["train"],
        compute_ngrams={"bigrams": True, "trigrams": True, "quadgrams": True},
        freq_filter=freq_filter,
    )

    collapsed = [y for x in data["train"]["cleaned_review_body"] for y in x]

    golden_ates = [
        y[0]
        for x in datasets.load_dataset(
            "json",
            data_files=[
                "data_as_jsonl/rest16/train.jsonl",
                "data_as_jsonl/rest15/train.jsonl",
                "data_as_jsonl/rest16/dev.jsonl",
                "data_as_jsonl/rest15/dev.jsonl",
            ],
        )["train"]["quads"]
        for y in x
    ]
    golden_ates = list(set(golden_ates))
    grouped_golden_ates = [
        list(g)
        for k, g in groupby(
            sorted(golden_ates, key=lambda x: len(x.split(" ")), reverse=True),
            key=lambda x: len(x.split(" ")),
        )
    ]
    (
        nouns_collapsed_counter,
        bigrams_df_filtered,
        trigrams_df_filtered,
        quadgrams_df_filtered,
    ) = get_candidate_ates(bigrams_df, trigrams_df, quadgrams_df, collapsed)

    matchers = []
    for group in grouped_golden_ates:
        matcher_golden_group = Matcher(nlp.vocab)
        for ate in group:
            matcher_golden_group.add(
                "ate", [[{"LOWER": x} for x in ate.lower().split(" ")]]
            )
        matchers.append(matcher_golden_group)

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
    for ate in [n[0].strip() for n in nouns_collapsed_counter[:5000]]:
        matcher_sw.add("ate", [[{"LEMMA": ate}]])
    matchers.append(matcher_sw)

    input_data = []
    for line in tqdm.tqdm(data["train"]["raw_text"]):
        sent_line = nltk.sent_tokenize(line)
        if split_long_sentences and len(sent_line) > 3:
            for s in sent_line:
                input_data.append(s)
        else:
            input_data.append(line)

    pool = multiprocessing.Pool(8)
    sharded = [
        input_data[(0 * 90_000) : (1 * 90_000)],
        input_data[(1 * 90_000) : (2 * 90_000)],
        input_data[(2 * 90_000) : (3 * 90_000)],
        input_data[(3 * 90_000) : (4 * 90_000)],
        input_data[(4 * 90_000) : (5 * 90_000)],
        input_data[(5 * 90_000) : (6 * 90_000)],
        input_data[(6 * 90_000) : (7 * 90_000)],
        input_data[(7 * 90_000) : (90 * 90_000)],
    ]
    sharded = [(x, matchers) for x in sharded]
    pool_result = pool.map(do_matching_on_subset, sharded)

    result = [
        (" ".join(y[0]), [[z, "", "", ""] for z in y[1]])
        for x in pool_result
        for y in x
    ]

    if skip_empty_extractions:
        result = [x for x in result if len(x[1]) > 0]

    if saving_path:
        with open(saving_path, "w+") as fout:
            for sentence, noisy_quads in tqdm.tqdm(result):
                _ = fout.write(sentence.strip())
                _ = fout.write("####")
                _ = fout.write("[")
                for quad in noisy_quads:
                    _ = fout.write(str(quad))
                    _ = fout.write(",")
                _ = fout.write("]")
                _ = fout.write("\n")

    return


if __name__ == "__main__":
    data = datasets.load_from_disk(
        "cache/yelp_academic_dataset_review_uf_6250_944b6ef7"
    )
    extract_ate(data)
