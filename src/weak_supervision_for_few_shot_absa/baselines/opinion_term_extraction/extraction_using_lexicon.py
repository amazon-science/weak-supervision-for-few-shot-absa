import json
import multiprocessing
import spacy
import tqdm


nlp = spacy.load("en_core_web_sm")


def extract_ote_from_sentence(sentence, matchers):
    doc = nlp(sentence)
    otes = []
    indices_in = set()
    for matcher in matchers:
        for _, start, end in matcher(doc):
            if start not in indices_in:
                otes.append((" ".join([x.text for x in doc[start:end]]), start))
                for i in range(start, end):
                    indices_in.add(i)
    otes = [x[0] for x in sorted(otes, key=lambda k: k[1])]
    if len(otes) > 0:
        return ([x.text for x in doc], otes)
    else:
        return ([x.text for x in doc], [])


def do_subset(input_data):
    (subset, (matcher_negated, matcher), skip_empty_extractions) = input_data
    keep_empty_extractions = not skip_empty_extractions
    result = []
    for line in tqdm.tqdm(subset):
        sentence, noisy_quads = line["sentence"], line["quads"]
        sentence_opinions = extract_ote_from_sentence(
            sentence, [matcher_negated, matcher]
        )
        for ot in sentence_opinions[1]:
            noisy_quads.append(["", "", "", ot])
        if keep_empty_extractions or len(sentence_opinions[1]) > 0:
            result.append((sentence, noisy_quads))
            # _=fout.write(sentence.strip())
            # _=fout.write("####")
            # _=fout.write('[')
            # for quad in noisy_quads:
            #     _=fout.write(str(quad))
            #     _=fout.write(',')
            # _=fout.write(']')
            # _=fout.write("\n")
    return result


"""
:param ate_annotated_file_path -> on what data to apply this function; should be of the same format as typical semeval absa data (so <sentence>####<quads>)
:param save_path               -> where to save the output
:param lexicon_path            -> where to find the lexicon which will be applied on this data
:param skip_empty_extractions  -> boolean flag for whether sentences where nothing was extracted should be included in the output or not

Extract opinion terms and append them to the quads; Note that the data is assumed to be with aspect terms already annotated
While this is not important for opinion term extraction, it is an artefact of early development of this project
The saving is done in the same semeval format

"""


def extract_ote(
    ate_annotated_file_path="logs/baselines/week6/ate_dataset/yelp_with_ates",
    save_path="logs/baselines/week6/ate_dataset/yelp_with_ates_otes",
    lexicon_path="data/lexicon/merged/lexicon.txt",
    skip_empty_extractions=True,
):
    with open(lexicon_path) as fin:
        lexicon = [x.strip() for x in fin.readlines()]

    from spacy.matcher import Matcher

    matcher = Matcher(nlp.vocab)
    matcher_negated = Matcher(nlp.vocab)

    for opinion_term in lexicon:
        matcher.add("ote", [[{"LOWER": opinion_term}]])
        matcher_negated.add("ote", [[{"LEMMA": "not"}, {"LOWER": opinion_term}]])
        matcher_negated.add("ote", [[{"LEMMA": "no"}, {"LOWER": opinion_term}]])

    data = []
    with open(ate_annotated_file_path) as fin:
        for line in fin:
            data.append(json.loads(line))
    step_size = len(data) // 8 + 1
    print(step_size)
    data_sharded = [
        data[(0 * step_size) : (1 * step_size)],
        data[(1 * step_size) : (2 * step_size)],
        data[(2 * step_size) : (3 * step_size)],
        data[(3 * step_size) : (4 * step_size)],
        data[(4 * step_size) : (5 * step_size)],
        data[(5 * step_size) : (6 * step_size)],
        data[(6 * step_size) : (7 * step_size)],
        data[(7 * step_size) :],
    ]
    data_sharded = [
        (x, (matcher_negated, matcher), skip_empty_extractions) for x in data_sharded
    ]
    pool = multiprocessing.Pool(8)
    pool_result = pool.map(do_subset, data_sharded)
    result = [{"sentence": y[0], "quads": y[1]} for x in pool_result for y in x]

    with open(save_path, "w+") as fout:
        for line in tqdm.tqdm(result):
            fout.write(f"{json.dumps(line)}\n")


if __name__ == "__main__":
    extract_ote()
