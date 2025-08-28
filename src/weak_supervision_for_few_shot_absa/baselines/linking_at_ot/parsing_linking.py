import spacy
import benepar
import datasets
from collections import defaultdict


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})


"""
This function will return the smallest element containing both the aspect term candidate and the opinion term candidate
The utility is in calculating the distance between the aspect term candidate and the opinion term candidate, as what we can
do is get the common ancestor, then calculate the distance between the common ancestor and each element
"""


def get_node_containing_tuple(
    aspect_term_candiate, opinion_term_candidate, parsed_sentence
):
    for child in parsed_sentence._.children:
        if (
            aspect_term_candiate.lower() in child.text.lower()
            and opinion_term_candidate.lower() in child.text.lower()
        ):
            return get_node_containing_tuple(
                aspect_term_candiate=aspect_term_candiate,
                opinion_term_candidate=opinion_term_candidate,
                parsed_sentence=child,
            )
    return parsed_sentence


def get_labels_with_node_containing_single(element, parsed_sentence) -> int:
    # assert(element in parsed_sentence.text) # sanity check
    for child in parsed_sentence._.children:
        if element in child.text:
            # print(child._.labels)
            labels = get_labels_with_node_containing_single(element, child)
            return labels + [child._.labels]
    return []


def get_distance_between_tuple(atc, otc, top_level_sentence):
    common_ancestor = get_node_containing_tuple(atc, otc, top_level_sentence)
    d1 = get_labels_with_node_containing_single(atc, common_ancestor)
    d2 = get_labels_with_node_containing_single(otc, common_ancestor)
    return d1[0] + d2[0]


def link_example(example, nlp):
    all_ates = [x[0] for x in example["quads"] if x[0] != ""]
    all_otes = [x[3] for x in example["quads"] if x[3] != ""]
    doc = nlp(example["sentence"])
    linking_result = link_example_preprocessed(doc, all_ates, all_otes)
    return [(x[0], "", "", x[1]) for x in linking_result]


def link_example_preprocessed(doc, all_ates, all_otes, max_distance=8):
    pairs = [(x, y) for x in all_ates for y in all_otes]
    result = []
    for sent in doc.sents:
        for atc, otc in pairs:
            if atc in sent.text and otc in sent.text:
                common_ancestor = get_node_containing_tuple(atc, otc, sent)

                labels_atc = get_labels_with_node_containing_single(
                    atc, common_ancestor
                )
                labels_otc = get_labels_with_node_containing_single(
                    otc, common_ancestor
                )
                labels_atc = [x[0] for x in labels_atc if x != ()]
                labels_otc = [x[0] for x in labels_otc if x != ()]

                distance_atc = len(labels_atc)
                distance_otc = len(labels_otc)

                # Some logic to do the linking
                # Long looking if conditions, but it all resumes to:
                # - if distance is 0 and the parent is NP
                # - if total distance is smaller than max, parent is S, atc is NP and otc is ADJP
                # The length comes from a couple of checks regarding the size of the lists/tuples
                if (
                    distance_atc == 0
                    and distance_otc == 0
                    and len(common_ancestor._.labels) > 0
                    and common_ancestor._.labels[0] == "NP"
                ):
                    result.append(
                        (atc, otc, distance_atc + distance_otc, labels_atc, labels_otc)
                    )
                elif (
                    distance_atc + distance_otc < max_distance
                    and len(common_ancestor._.labels) > 0
                    and common_ancestor._.labels[0] == "S"
                    and len(labels_atc) > 0
                    and len(labels_otc) > 0
                    and labels_atc[0] == "NP"
                    and labels_otc[0] == "ADJP"
                    and "S" not in labels_atc
                    and "S" not in labels_otc
                ):
                    if atc not in " ".join(
                        [x[0] for x in result]
                    ):  # prevent situations like "french fries were very good and the fries were crispy" -> (french fries, good), (fries, good) -- i.e. one ate is part of a larger ate, but they appear separately
                        result.append(
                            (
                                atc,
                                otc,
                                distance_atc + distance_otc,
                                labels_atc,
                                labels_otc,
                            )
                        )
                    else:
                        if otc not in [x[1] for x in result]:
                            result.append(
                                (
                                    atc,
                                    otc,
                                    distance_atc + distance_otc,
                                    labels_atc,
                                    labels_otc,
                                )
                            )
    return result


import pickle
import tqdm
import json

with open(
    "logs/baselines/week7/linking/constituency_parsing/constituency_data_preprocessed.pkl",
    "rb",
) as fin:
    data = pickle.load(fin)
print(len(data))
exit()

result = []
for d in tqdm.tqdm(data):
    (data_dict, doc) = d
    all_ates = [x[0] for x in data_dict["quads"] if x[0] != ""]
    all_otes = [x[3] for x in data_dict["quads"] if x[3] != ""]
    linking_result = link_example_preprocessed(doc, all_ates, all_otes)
    linking_result = [(x[0], "", "", x[1]) for x in linking_result]
    result.append({"sentence": data_dict["sentence"], "quads": linking_result})

with open(
    "logs/baselines/week7/linking/constituency_parsing/dataset_ate_ote_linked.jsonl",
    "w+",
) as fout:
    for line in result:
        if len(line["quads"]) > 0:
            fout.write(f"{json.dumps(line)}\n")
