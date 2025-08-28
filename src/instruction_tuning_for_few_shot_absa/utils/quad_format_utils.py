"""
Various utilities related to the initial quad format
This includes:
- writing to file
- converting to huggingface dataset
- converting to quad format

There is some commented code left. This was intentional, to have
some sort of history. They were used to convert from few-shot datasets
to the huggingface dataset format (.jsonl)
The scripts used to sample for k-shot learning gives us files in `.txt` format
"""

import json
import tqdm


def write_sentence_quad_to_file(sentence_quads, fout):
    (sentence, noisy_quads) = sentence_quads
    _ = fout.write(sentence.strip())
    _ = fout.write("####")
    _ = fout.write("[")
    for quad in noisy_quads:
        _ = fout.write(str(quad))
        _ = fout.write(",")
    _ = fout.write("]")
    _ = fout.write("\n")


# Converts the format with `<sentence>####<quads>` to jsonl format with {'sentence': <..>, 'quads': <..>}
def convert_to_huggingface_dataset(input_path, output_path):
    data = []
    with open(input_path) as fin:
        for line in tqdm.tqdm(fin.readlines()):
            sentence, quad = line.split("####")
            quad = eval(quad)
            data.append({"sentence": sentence, "quads": quad})

    with open(output_path, "w+") as fout:
        for line in tqdm.tqdm(data):
            _ = fout.write(json.dumps(line))
            _ = fout.write("\n")

    return data


# Converts the jsonl format {'sentence': <..>, 'quads': <..>} to `<sentence>####<quads>`
def convert_to_quad_dataset(input_path, output_path):
    import datasets

    data = datasets.load_dataset("json", data_files=input_path)["train"]
    with open(output_path, "w+") as fout:
        for line in tqdm.tqdm(data):
            sentence = line["sentence"]
            quads = str(line["quads"])
            _ = fout.write(f"{sentence}####{quads}\n")
        return data
