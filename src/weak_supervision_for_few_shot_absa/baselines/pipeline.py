from typing import Dict, List

import spacy
import datasets
from src.weak_supervision_for_few_shot_absa.baselines.adding_sentiment.predict_sentiment import (
    add_sentiment,
)
from src.weak_supervision_for_few_shot_absa.baselines.linking_at_ot.nli_linking import (
    do_nli_link,
)

from src.weak_supervision_for_few_shot_absa.baselines.utils import (
    map_clean_comments,
)
from src.weak_supervision_for_few_shot_absa.baselines.aspect_term_extraction.most_frequent_nouns import (
    extract_ate,
)
from src.weak_supervision_for_few_shot_absa.baselines.opinion_term_extraction.extraction_using_lexicon import (
    extract_ote,
)
from src.weak_supervision_for_few_shot_absa.utils.quad_format_utils import (
    convert_to_huggingface_dataset,
)


"""
Pipeline for the noisy annotate
This function contains the necessary code to perform the complete annotation:
- aspect term extraction
- opinion term extraction
- linking aspect terms with their corresponding opinion terms
- adding sentiment to the (aspect term, opinion term) tuple

There order is (left to right pseudo image below):
- aspect term extraction  -> 
                             -> linking -> adding sentiment
- opinion term extraction ->

It receives as parameter a complete dataset. It cannot work on a per-sentence basis
because the aspect term extraction needs full corpus statistics; It would be possible after
refactoring to give the full corpus statistics needed for aspect term extraction as parameter,
but this is not implemented
It receives as parameter a dictionary, containing the necessary information for each step in the pipeline:
{
    'do_ate'             : bool,
    'do_ote'             : bool,
    'do_ate_ote_link'    : bool,
    'do_sent'            : bool,
    'ate_config'         : dict,
    'ote_config'         : dict,
    'ate_ote_link_config': dict,
    'sent_config'        : dict,
}

:param dataset   -> on what data to apply this function; should be a huggingface datasets type of data (so list of dictionaries)
:param save_path -> where to save the result
:param config    -> the configuration dictionary, which encompass configs for all the steps to be applied

"""


def noisy_annotate_pipeline(hf_dataset: str, save_path: str, config: Dict):

    if config["do_ate"]:
        print("Doing ATE..")
        nlp = spacy.load("en_core_web_sm")
        data = hf_dataset.map(
            lambda x: {
                "cleaned_review_body": map_clean_comments(
                    x[config["ate_config"]["text_column_name"]], nlp
                )
            },
            batched=True,
            batch_size=5000,
        )
        ate_save_path = config.get("save_ate_path", "/tmp/data_with_ate.jsonl")
        extract_ate(data, ate_save_path, **config["ate_config"])
    if config["do_ote"]:
        print("Doing OTE..")
        ate_save_path = config.get("save_ate_path", "/tmp/data_with_ate.jsonl")
        ote_save_path = config.get("save_ote_path", "/tmp/data_with_ate_ote.jsonl")
        extract_ote(ate_save_path, ote_save_path, **config["ote_config"])
        # convert_to_huggingface_dataset('/tmp/data_with_ate_ote.txt', '/tmp/data_with_ate_ote.jsonl')
    if config["do_ate_ote_link"]:
        print("Doing NLI Link..")
        ote_save_path = config.get("save_ote_path", "/tmp/data_with_ate_ote.jsonl")
        ate_ote_linked_save_path = config.get(
            "save_ate_ote_linked_save_path", "/tmp/data_with_ate_ote_linked.jsonl"
        )
        do_nli_link(
            ote_save_path,
            save_path=ate_ote_linked_save_path,
            **config["ate_ote_link_config"]
        )
    if config["do_sent"]:
        print("Adding sentiment..")
        ate_ote_linked_save_path = config.get(
            "save_ate_ote_linked_save_path", "/tmp/data_with_ate_ote_linked.jsonl"
        )
        add_sentiment(
            ate_ote_linked_save_path, save_path=save_path, **config["sent_config"]
        )


def do_amazon_electronics():
    config = {
        "do_ate": True,
        "do_ote": False,
        "do_ate_ote_link": False,
        "do_sent": False,
        "save_ate_path": "logs/baselines/week14/test/data_with_ate.jsonl",
        "save_ote_path": "logs/baselines/week14/test/data_with_ate_ote.jsonl",
        "save_ate_ote_linked_save_path": "logs/baselines/week14/test/data_with_ate_ote_linked_nli.jsonl",
        "ate_config": {
            # 'freq_filter'               : {'bigram': 1, 'trigram': 1, 'quadgram': 1},
            "split_long_sentences": True,
            "skip_empty_extractions": True,
            "text_column_name": "review_body",
            # 'take_top_n_nouns'          : 10000, # Total number of nouns: 55582
        },
        "ote_config": {
            "skip_empty_extractions": True,
        },
        "ate_ote_link_config": {"threshold": 0.75},
        "sent_config": {"threshold": 0.75, "sent_classifier_type": "pos_neg"},
        # 'sent_config'        : {'threshold': 0.5, 'sent_classifier_type': 'pos_neg_neutral'},
    }
    from datasets import DatasetDict

    noisy_annotate_pipeline(
        DatasetDict(
            {
                "train": datasets.load_dataset("amazon_us_reviews", "Electronics_v1_00")
                .shuffle(seed=1)["train"]
                .select(range(100000))
            }
        ),
        "<ABSAWithWeaklySupervisedPreTraining_Path>/logs/baselines/week14/amazon_electronics/electronics_dataset.jsonl",
        config,
    )


def do_yelp():
    config = {
        "do_ate": True,
        "do_ote": True,
        "do_ate_ote_link": True,
        "do_sent": False,
        "save_ate_path": "logs/baselines/week14/test/data_with_ate.jsonl",
        "save_ote_path": "logs/baselines/week14/test/data_with_ate_ote.jsonl",
        "save_ate_ote_linked_save_path": "logs/baselines/week14/test/data_with_ate_ote_linked_nli.jsonl",
        "ate_config": {
            # 'freq_filter'           : {'bigram': 1, 'trigram': 1, 'quadgram': 1},
            "split_long_sentences": True,
            "skip_empty_extractions": True,
            "text_column_name": "raw_text",
            # 'take_top_n_nouns'      : 10000, # 54553
        },
        "ote_config": {
            "skip_empty_extractions": True,
        },
        "ate_ote_link_config": {"threshold": 0.75},
        "sent_config": {"threshold": 0.75, "sent_classifier_type": "pos_neg"},
        # 'sent_config'        : {'threshold': 0.5, 'sent_classifier_type': 'pos_neg_neutral'},
    }
    noisy_annotate_pipeline(
        datasets.load_dataset(
            "json",
            data_files=["logs/baselines/yelp_academic_dataset_review_uf_6250.json"],
        ),
        "<ABSAWithWeaklySupervisedPreTraining_Path>/logs/baselines/week14/test/yelp_dataset.jsonl",
        config,
    )


def do_zeroshot_baseline():
    base_config = {
        "do_ate": True,
        "do_ote": True,
        "do_ate_ote_link": True,
        "do_sent": True,
        "ate_config": {
            "freq_filter": {"bigram": 1, "trigram": 1, "quadgram": 1},
            "split_long_sentences": False,
            "skip_empty_extractions": True,
            "text_column_name": "sentence",
            "take_top_n_nouns": 200,  # 903
        },
        "ote_config": {
            "skip_empty_extractions": True,
        },
        "ate_ote_link_config": {"threshold": 0.75, "skip_long_sentences": False},
        "sent_config": {"threshold": 0.50, "sent_classifier_type": "pos_neg"},
    }
    for dataset_path, dataset_savepath, top_n_nouns in [
        (
            "data_as_jsonl/lap14/dev.jsonl",
            "/tmp/data_with_ate_ote_sent_lap14.jsonl",
            90,
        ),  # 903 total nouns; use top 10% (to match the dataset creation)
        (
            "data_as_jsonl/rest15/dev.jsonl",
            "/tmp/data_with_ate_ote_sent_rest15.jsonl",
            77,
        ),  # 770 total nouns; use top 10% (to match the dataset creation)
        (
            "data_as_jsonl/rest16/dev.jsonl",
            "/tmp/data_with_ate_ote_sent_rest16.jsonl",
            95,
        ),  # 953 total nouns; use top 10% (to match the dataset creation)
    ]:
        config = {**base_config}
        config["ate_config"]["take_top_n_nouns"] = top_n_nouns
        print(dataset_path)
        print(config)
        noisy_annotate_pipeline(
            datasets.load_dataset("json", data_files=[dataset_path]),
            dataset_savepath,
            config,
        )
        pred_data = datasets.load_dataset("json", data_files=[dataset_savepath])[
            "train"
        ]
        gold_data = datasets.load_dataset("json", data_files=dataset_path)["train"]

        pred_sent_to_quads = {x["sentence"]: x["quads"] for x in pred_data}

        from src.weak_supervision_for_few_shot_absa.instruction_tuning.eval_utils import (
            compute_f1_scores,
        )

        gold = []
        # TASK4
        print("TASK4")
        for quads in gold_data["quads"]:
            gold.append([[q[0], "", q[2], q[3]] for q in quads])
        pred = []
        for sent in gold_data["sentence"]:
            pred.append(pred_sent_to_quads.get(sent, []))
        print(compute_f1_scores(pred, gold))

        # TASK2
        print("TASK2")
        gold = []
        for quads in gold_data["quads"]:
            gold.append([[q[0], "", q[2], ""] for q in quads])
        pred = []
        for sent in gold_data["sentence"]:
            pred_quads = pred_sent_to_quads.get(sent, [])
            if len(pred_quads) > 0:
                pred.append([[q[0], "", q[2], ""] for q in pred_quads])
            else:
                pred.append([])

        print(compute_f1_scores(pred, gold))

        # TASK1
        print("TASK1")
        gold = []
        for quads in gold_data["quads"]:
            gold.append([[q[0], "", "", ""] for q in quads])
        pred = []
        for sent in gold_data["sentence"]:
            pred_quads = pred_sent_to_quads.get(sent, [])
            if len(pred_quads) > 0:
                pred.append([[q[0], "", "", ""] for q in pred_quads])
            else:
                pred.append([])

        print(compute_f1_scores(pred, gold))

        # ATE + OTE Pred
        print("ATE + OTE")
        gold = []
        for quads in gold_data["quads"]:
            gold.append([[q[0], "", q[2], ""] for q in quads])
        pred = []
        for sent in gold_data["sentence"]:
            pred_quads = pred_sent_to_quads.get(sent, [])
            if len(pred_quads) > 0:
                pred.append([[q[0], "", q[2], ""] for q in pred_quads])
            else:
                pred.append([])

        print(compute_f1_scores(pred, gold))

        # OTE Pred
        print("OTE")
        gold = []
        for quads in gold_data["quads"]:
            gold.append([["", "", q[2], ""] for q in quads])
        pred = []
        for sent in gold_data["sentence"]:
            pred_quads = pred_sent_to_quads.get(sent, [])
            if len(pred_quads) > 0:
                pred.append([["", "", q[2], ""] for q in pred_quads])
            else:
                pred.append([])

        print(compute_f1_scores(pred, gold))
        print("----------------\n\n\n")


# python -m src.weak_supervision_for_few_shot_absa.baselines.pipeline
if __name__ == "__main__":
    # do_amazon_electronics()
    # do_yelp()
    do_zeroshot_baseline()
    # exit()
