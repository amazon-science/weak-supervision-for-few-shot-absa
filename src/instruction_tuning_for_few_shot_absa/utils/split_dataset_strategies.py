"""
The idea of this script is to split the data into disjoint sets
There can be multiple ways of doing this, depending on how strict
we want "disjoint" to be
See the comments for each function
"""

import tqdm
import random
import json
import datasets
from collections import defaultdict, Counter

# from src.absa_with_weak_supervision.utils.quad_format_utils import write_sentence_quad_to_file

"""
Only append to test  when ANY the ates in it are from the sampled test list
Only append to train when NONE of the ates in it are from the sampled test list
chicken breast; salad (TEST)
lobster, sandwich (TRAIN)

I had chicken breast and a sandwich -> DISCARD
I had chicken breast -> TEST
I had lobster and a sandwich -> TRAIN
"""


def relaxed_sampling(ates, how_many):
    ates_sampled = set(random.sample(list(ates.keys()), how_many))
    # otes_sampled = []
    train = []
    test = []
    for sentence, quad in tqdm.tqdm(data):
        ats_in_quad = [x[0] for x in quad if x[0] != ""]
        if any(x in ates_sampled for x in ats_in_quad):
            test.append((sentence, quad))
        else:
            train.append((sentence, quad))

    return train, test


"""
Only append to test when ALL the ates in it are from the sampled test list
Only append to train when NONE of the ates in it are from the sampled test list
"""


def strict_sampling(data, how_many):
    # random.seed(1)
    ates = set([y[0] for x in data["quads"] for y in x])
    otes = set([y[3] for x in data["quads"] for y in x])
    ates_sampled = set(random.sample(list(ates.keys()), how_many))
    otes_sampled = set(random.sample(list(ates.keys()), how_many))
    # otes_sampled = []
    train = []
    test = []
    for sentence, quad in tqdm.tqdm(data):
        ats_in_quad = [x[0] for x in quad if x[0] != ""]
        if all(x in ates_sampled for x in ats_in_quad):
            test.append((sentence, quad))

        if not any(x in ates_sampled for x in ats_in_quad):
            train.append((sentence, quad))

    return train, test


def strict_sampling_variant():
    data = []
    ates = defaultdict(int)
    otes = defaultdict(int)
    with open("logs/baselines/week6/ate_dataset/yelp_with_ates_otes") as fin:
        for line in tqdm.tqdm(fin):
            (sentence, quad) = line.split("####")
            quad = eval(quad)
            data.append((sentence.strip(), quad))

            for at, ac, s, ot in quad:
                if at != "":
                    ates[at.lower()] += 1
                if ot != "":
                    otes[ot.lower()] += 1

    random.seed(1)
    # train, test = relaxed_sampling(ates, 1100)
    train, test = strict_sampling(ates, 2500)

    print(len(train))
    print(len(test))
    print(len(train) / (len(train) + len(test)))
    print(len(data))


# A simple, random split
# Works for the jsonl format
def simple_split():
    import datasets

    data = datasets.load_dataset(
        "json", data_files="logs/baselines/week6/ate_dataset/yelp_with_ates_otes.jsonl"
    )["train"].train_test_split(test_size=0.1, seed=1)

    with open(
        "logs/baselines/week6/ate_dataset/yelp_with_ates_otes_train.jsonl", "w+"
    ) as fin:
        for line in data["train"]:
            _ = fin.write(json.dumps(line))
            _ = fin.write("\n")

    with open(
        "logs/baselines/week6/ate_dataset/yelp_with_ates_otes_test.jsonl", "w+"
    ) as fin:
        for line in data["test"]:
            _ = fin.write(json.dumps(line))
            _ = fin.write("\n")


# PYTHONHASHSEED=1 python -m src.absa_with_weak_supervision.utils.split_dataset_strategies
# The PYTHONHASHSEED ensures that the elements appear in the same order in the set between runs
if __name__ == "__main__":
    data_yelp = datasets.load_dataset(
        "json", data_files=["logs/baselines/week14/yelp/yelp_dataset.jsonl"]
    )["train"]
    data_elec = datasets.load_dataset(
        "json",
        data_files=[
            "logs/baselines/week14/amazon_electronics/electronics_dataset.jsonl"
        ],
    )["train"]

    ates_yelp = sorted(
        Counter([y[0].lower() for x in data_yelp["quads"] for y in x]).items(),
        key=lambda x: x[1],
    )
    otes_yelp = sorted(
        Counter([y[3].lower() for x in data_yelp["quads"] for y in x]).items(),
        key=lambda x: x[1],
    )
    ates_elec = sorted(
        Counter([y[0].lower() for x in data_elec["quads"] for y in x]).items(),
        key=lambda x: x[1],
    )
    otes_elec = sorted(
        Counter([y[3].lower() for x in data_elec["quads"] for y in x]).items(),
        key=lambda x: x[1],
    )
    print(len(ates_yelp))
    print(len(otes_yelp))
    print(len(ates_elec))
    print(len(otes_elec))

    # We tried multiple seeds to see which one produces a splits of reasonable size
    # This is needed because, if unlucky, one could end up with an aspect term
    # which is in the majority of the examples (extreme situation), which makes
    # splitting hard to do
    for seed in [
        # 1, 2, 3, 4, 5, 6, 7, 8, 9,
        # 10, 20, 30, 40, 50, 60, 70, 80, 90,
        # 77894, 30134, 55889, 74624, 11561, 99766, 143, 83438, 20230, 96959,
        # 92758, 34666, 72180, 29504, 13494, 55322, 699, 63611, 91924, 87840, 12850, 43104, 86008, 90409, 51528, 33593, 56741, 3180, 27720, 39935, 57289, 61272, 5011, 45871, 2130, 51686, 44350, 66412, 7620, 53044, 58502, 88687, 78517, 15053, 7871, 21609, 59512, 65462, 592, 55959, 58424, 5312, 87327, 41870, 77552, 23852, 61479, 14645, 14885, 98027
        20,
    ]:
        random.seed(seed)
        # We do the splitting by selecting 1400 unfrequent ones and 700 frequent ones for aspect terms
        # Similarly for opinion terms
        ates_yelp_sampled = set(
            list(
                set(random.sample([x[0] for x in ates_yelp if x[1] <= 10], 1400)).union(
                    random.sample([x[0] for x in ates_yelp if not (x[1] <= 10)], 700)
                )
            )
        )
        otes_yelp_sampled = set(
            list(
                set(random.sample([x[0] for x in otes_yelp if x[1] <= 10], 700)).union(
                    random.sample([x[0] for x in otes_yelp if not (x[1] <= 10)], 350)
                )
            )
        )
        ates_elec_sampled = set(
            list(
                set(random.sample([x[0] for x in ates_elec if x[1] <= 10], 1400)).union(
                    random.sample([x[0] for x in ates_elec if not (x[1] <= 10)], 700)
                )
            )
        )
        otes_elec_sampled = set(
            list(
                set(random.sample([x[0] for x in otes_elec if x[1] <= 10], 700)).union(
                    random.sample([x[0] for x in otes_elec if not (x[1] <= 10)], 350)
                )
            )
        )

        ates_sampled = set(ates_yelp_sampled).union(set(ates_elec_sampled))
        otes_sampled = set(otes_yelp_sampled).union(set(otes_elec_sampled))
        data = datasets.concatenate_datasets([data_yelp, data_elec])
        train2_yelp = []
        train2_elec = []
        test2_yelp = []
        test2_elec = []
        for line in data_yelp:
            sentence = line["sentence"]
            quad = line["quads"]

            ats_in_quad = [x[0] for x in quad if x[0] != ""]
            ots_in_quad = [x[3] for x in quad if x[3] != ""]
            ats_ots_zipped = list(zip(ats_in_quad, ots_in_quad))

            if all(
                (
                    x[0].lower() in ates_yelp_sampled
                    and x[1].lower() in otes_yelp_sampled
                )
                for x in ats_ots_zipped
            ):
                test2_yelp.append(line)
            if not any(
                (x[0].lower() in ates_yelp_sampled or x[1].lower() in otes_yelp_sampled)
                for x in ats_ots_zipped
            ):
                train2_yelp.append(line)

        for line in data_elec:
            sentence = line["sentence"]
            quad = line["quads"]

            ats_in_quad = [x[0] for x in quad if x[0] != ""]
            ots_in_quad = [x[3] for x in quad if x[3] != ""]
            ats_ots_zipped = list(zip(ats_in_quad, ots_in_quad))

            if all(
                (x[0].lower() in ates_elec_sampled and x[1] in otes_elec_sampled)
                for x in ats_ots_zipped
            ):
                test2_elec.append(line)
            if not any(
                (x[0].lower() in ates_elec_sampled or x[1] in otes_elec_sampled)
                for x in ats_ots_zipped
            ):
                train2_elec.append(line)

        print("\n-----------")
        print(seed)
        print(len(train2_elec + train2_yelp))
        print(len(test2_elec + test2_yelp))
        print("-----------\n")

        train2 = train2_yelp + train2_elec
        test2 = test2_yelp + test2_elec

        with open(
            "logs/baselines/week14/amazon_nli_linked_and_sentiment_train.jsonl", "w+"
        ) as fout:
            for line in train2_elec:
                _ = fout.write(f"{json.dumps(line)}\n")
        with open(
            "logs/baselines/week14/amazon_nli_linked_and_sentiment_test.jsonl", "w+"
        ) as fout:
            for line in test2_elec:
                _ = fout.write(f"{json.dumps(line)}\n")
        with open(
            "logs/baselines/week14/yelp_nli_linked_and_sentiment_train.jsonl", "w+"
        ) as fout:
            for line in train2_yelp:
                _ = fout.write(f"{json.dumps(line)}\n")
        with open(
            "logs/baselines/week14/yelp_nli_linked_and_sentiment_test.jsonl", "w+"
        ) as fout:
            for line in test2_yelp:
                _ = fout.write(f"{json.dumps(line)}\n")
        with open(
            "logs/baselines/week12/nli_linked_and_sentiment_train.jsonl", "w+"
        ) as fout:
            for line in train2:
                _ = fout.write(f"{json.dumps(line)}\n")
        with open(
            "logs/baselines/week12/nli_linked_and_sentiment_test.jsonl", "w+"
        ) as fout:
            for line in test2:
                _ = fout.write(f"{json.dumps(line)}\n")

    # simple_split()
    # strict_sampling_variant()
