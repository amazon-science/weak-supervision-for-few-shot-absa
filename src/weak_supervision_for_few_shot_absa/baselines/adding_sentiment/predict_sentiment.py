import json
import tqdm
import datasets
import pandas as pd
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset


"""
Add positive negative sentiment
"""


def add_pos_neg_sentiment(data_path, threshold):
    sentiment_pipeline = pipeline("sentiment-analysis", device=0)
    sentiment_pipeline(["I love you", "I hate you"])
    data = datasets.load_dataset("json", data_files=data_path)

    quads = [f"{q[0]} is {q[3]}" for quad in data["train"]["quads"] for q in quad]
    outputs = []
    for out in tqdm.tqdm(
        sentiment_pipeline(
            KeyDataset(
                datasets.Dataset.from_pandas(
                    pd.DataFrame(data=[{"text": q} for q in quads])
                ),
                "text",
            ),
            batch_size=128,
        )
    ):
        outputs.append(out)

    i = 0
    new_data = []
    for line in tqdm.tqdm(data["train"], total=len(quads)):
        new_quads = []
        for quad in line["quads"]:
            sent = outputs[i]
            i += 1
            if sent["score"] > threshold:
                new_quads.append([quad[0], "", sent["label"].lower(), quad[3]])
        if len(new_quads) > 0:
            new_data.append({"sentence": line["sentence"], "quads": new_quads})

    return new_data


"""
Add positive, negative, neutral sentiment
"""


def add_pos_neg_neutral_sentiment(data_path, threshold):
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
        return_all_scores=True,
        device=0,
    )
    mapping = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    sentiment_pipeline(["I love you", "I hate you"])
    data = datasets.load_dataset("json", data_files=data_path)

    quads = [f"{q[0]} is {q[3]}" for quad in data["train"]["quads"] for q in quad]
    outputs = []
    for out in tqdm.tqdm(
        sentiment_pipeline(
            KeyDataset(
                datasets.Dataset.from_pandas(
                    pd.DataFrame(data=[{"text": q} for q in quads])
                ),
                "text",
            ),
            batch_size=128,
        )
    ):
        outputs.append(out)

    i = 0
    new_data = []
    for line in tqdm.tqdm(data["train"], total=len(quads)):
        new_quads = []
        for quad in line["quads"]:
            pred_sentiment = outputs[i]
            i += 1
            for sentiment in pred_sentiment:
                if sentiment["score"] > threshold:
                    new_quads.append(
                        [quad[0], "", mapping[sentiment["label"]], quad[3]]
                    )
        if len(new_quads) > 0:
            new_data.append({"sentence": line["sentence"], "quads": new_quads})

    return new_data


"""
:param data_path -> on what data to apply this function; should be a huggingface datasets type of data (so list of dictionaries)
:param save_path -> where to save the result
"""


def add_sentiment(data_path, save_path, threshold=0.75, sent_classifier_type="pos_neg"):
    if sent_classifier_type == "pos_neg":
        new_data = datasets.Dataset.from_pandas(
            pd.DataFrame(data=add_pos_neg_sentiment(data_path, threshold))
        )
    elif sent_classifier_type == "pos_neg_neutral":
        new_data = datasets.Dataset.from_pandas(
            pd.DataFrame(data=add_pos_neg_neutral_sentiment(data_path, threshold))
        )
    else:
        raise ValueError(
            f"Unknown type {sent_classifier_type}. It has to be in [`pos_neg`, `pos_neg_neutral`]"
        )

    with open(save_path, "w+") as fout:
        for line in new_data:
            _ = fout.write(f"{json.dumps(line)}\n")
