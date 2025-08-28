"""
Apply the methods implemented in:
- src.weak_supervision_for_few_shot_absa.baselines.aspect_term_extraction
- src.weak_supervision_for_few_shot_absa.baselines.opinion_term_extraction
On the (annotated) SemEval dev partition
The goal is to see how many aspect terms/opinion terms are captured
"""

import spacy
import datasets
from src.weak_supervision_for_few_shot_absa.baselines.opinion_term_extraction.extraction_using_lexicon import (
    extract_ote,
)

from src.weak_supervision_for_few_shot_absa.baselines.utils import (
    map_clean_comments,
)
from src.weak_supervision_for_few_shot_absa.baselines.aspect_term_extraction.most_frequent_nouns import (
    build_ngrams,
    extract_ate,
)
from src.weak_supervision_for_few_shot_absa.instruction_tuning.eval_utils import (
    compute_f1_scores,
)


def apply_on_semeval(semeval_dataset, ate_save_path, ate_ote_save_path):
    nlp = spacy.load("en_core_web_sm")

    extract_ate(
        semeval_dataset,
        ate_save_path,
        {"bigram": 1, "trigram": 1, "quadgram": 1},
        split_long_sentences=False,
        skip_empty_extractions=False,
        text_column_name="sentence",
    )
    extract_ote(ate_save_path, ate_ote_save_path, skip_empty_extractions=False)


if __name__ == "__main__":
    # for (data_path, ate_save_path, ate_ote_save_path) in [
    # ('data_as_jsonl/rest15/dev.jsonl', 'data2/rest15_dev.txt', 'data2/rest15_dev_atot.txt'),
    # ('data_as_jsonl/rest16/dev.jsonl', 'data2/rest16_dev.txt', 'data2/rest16_dev_atot.txt'),
    # ('data_as_jsonl/lap14/dev.jsonl', 'data2/lap14_dev.txt', 'data2/lap14_dev_atot.txt'),
    # ]:
    (data_path, ate_save_path, ate_ote_save_path) = (
        "data_as_jsonl/rest16/dev.jsonl",
        "data2/rest16_dev.txt",
        "data2/rest16_dev_atot.txt",
    )
    nlp = spacy.load("en_core_web_sm")
    data = datasets.load_dataset("json", data_files=data_path).map(
        lambda x: {"cleaned_review_body": map_clean_comments(x["sentence"], nlp)},
        batched=True,
        batch_size=5000,
    )
    apply_on_semeval(data, ate_save_path, ate_ote_save_path)

    with open(ate_ote_save_path) as fin:
        noisy_data = []
        for line in fin:
            sentence, quads = line.split("####")
            quads = eval(quads)
            noisy_data.append(quads)

    gold_ats = [[y[0] for y in x] for x in data["train"]["quads"]]
    pred_ats = [[y[0] for y in x if y[0] != ""] for x in noisy_data]
    pred_ats = [["NULL"] if x == [] else x for x in pred_ats]

    gold_ots = [[y[-1] for y in x] for x in data["train"]["quads"]]
    pred_ots = [[y[-1] for y in x if y[-1] != ""] for x in noisy_data]

    print("-" * 100)
    print(data_path)
    print(compute_f1_scores(pred_ats, gold_ats))
    print(compute_f1_scores(pred_ots, gold_ots))
    print("-" * 100)
