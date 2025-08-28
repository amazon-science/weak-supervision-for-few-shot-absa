from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import datasets
import tqdm
import json
import pandas as pd
from transformers.pipelines.pt_utils import KeyDataset


def nli_link(example, model, tokenizer, threshold=0.5):
    all_ates = [x[0] for x in example["quads"] if x[0] != ""]
    all_otes = [x[3] for x in example["quads"] if x[3] != ""]
    pairs = [(x, y, f"{x} is {y}") for x in all_ates for y in all_otes]

    input_pairs = [(example["sentence"], pair[2]) for pair in pairs]

    inputs = tokenizer(
        ["</s></s>".join(input_pair) for input_pair in input_pairs],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=250,
    )
    with torch.no_grad():
        logits = model(**{k: v.to(model.device) for (k, v) in inputs.items()}).logits
        probs = torch.softmax(logits, dim=1).tolist()

    quads = []
    for i, nli_result in enumerate(probs):
        if nli_result[0] > threshold:
            quads.append([pairs[i][0], "", "", pairs[i][1]])

    return {"sentence": example["sentence"], "quads": quads}


"""
:param data_path  -> on what data to apply this function; should be a huggingface datasets type of data (so list of dictionaries)
:param model_name -> the name of the model which will do the NLI linking
:param save_path  -> where to save the result
:param threshold  -> NLI confidence threshold

Note that this function does the skipping by default. So the final data will not contain sentences with empty quads

"""


def do_nli_link(
    data_path="logs/baselines/week6/ate_dataset/yelp_with_ates_otes_train.jsonl",
    model_name="symanto/mpnet-base-snli-mnli",
    save_path="logs/baselines/week7/linking/nli/dataset_ate_ote_linked.jsonl",
    threshold=0.75,
    length_threshold=100,
    skip_long_sentences=True,
):
    data = (
        datasets.load_dataset("json", data_files=data_path)["train"]
        .map(lambda x: {"length": len(x["sentence"].split(" "))})
        .sort("length")
    )  # .remove_columns('length')
    if skip_long_sentences:
        data = data.filter(lambda x: x["length"] < length_threshold)
    else:
        data = data.remove_columns("length")

    tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 300}
    nli_pipeline = pipeline(model=model_name, tokenizer=model_name, device=0)

    nli_data = []
    for example in tqdm.tqdm(data):
        all_ates = [x[0] for x in example["quads"] if x[0] != ""]
        all_otes = [x[3] for x in example["quads"] if x[3] != ""]
        pairs = [(x, y, f"{x} is {y}") for x in all_ates for y in all_otes]
        input_pairs = [(example["sentence"], pair[2]) for pair in pairs]
        nli_data += ["</s></s>".join(input_pair) for input_pair in input_pairs]
        # inputs = tokenizer(["</s></s>".join(input_pair) for input_pair in input_pairs], return_tensors="pt", truncation=True, padding=True, max_length=250)

    outputs = []
    for out in tqdm.tqdm(
        nli_pipeline(
            KeyDataset(
                datasets.Dataset.from_pandas(
                    pd.DataFrame(data=[{"text": q} for q in nli_data])
                ),
                "text",
            ),
            batch_size=64,
            **tokenizer_kwargs,
        ),
        total=len(nli_data),
    ):
        outputs.append(out)

    i = 0
    new_data = []
    for example in tqdm.tqdm(data):
        all_ates = [x[0] for x in example["quads"] if x[0] != ""]
        all_otes = [x[3] for x in example["quads"] if x[3] != ""]
        pairs = [(x, y, f"{x} is {y}") for x in all_ates for y in all_otes]
        input_pairs = [(example["sentence"], pair[2]) for pair in pairs]
        new_quads = []
        for ate, ote, _ in pairs:
            nli_output = outputs[i]
            i += 1
            if nli_output["label"] == "ENTAILMENT" and nli_output["score"] > threshold:
                new_quads.append([ate, "", "", ote])
        new_data.append({"sentence": example["sentence"], "quads": new_quads})

    with open(save_path, "w+") as fout:
        for result in new_data:
            if len(result["quads"]) > 0:
                fout.write(f"{json.dumps(result)}\n")

    # model = AutoModelForSequenceClassification.from_pretrained(model_name).to(torch.device('cuda'))
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # data = datasets.load_dataset('json', data_files=data_path)['train']
    # process_long_sentences = not skip_long_sentences
    # with open(save_path, 'w+') as fout:
    #     data_short = [x for x in data if len(x['sentence'].split(' ')) < length_threshold]
    #     for example in tqdm.tqdm(data_short):
    #         result = nli_link(example, model, tokenizer, threshold)
    #         if len(result['quads']) > 0:
    #             fout.write(f'{json.dumps(result)}\n')
    #     if process_long_sentences:
    #         data_long = [x for x in data if not len(x['sentence'].split(' ')) < length_threshold]
    #         print(len(data_long))
    #         model = model.to(torch.device('cpu'))
    #         for example in tqdm.tqdm(data_long):
    #             result = nli_link(example, model, tokenizer, threshold)
    #             if len(result['quads']) > 0:
    #                 fout.write(f'{json.dumps(result)}\n')
