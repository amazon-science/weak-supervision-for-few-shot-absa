import datasets
import gensim
import tqdm
import nltk
from gensim.test.utils import datapath
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """

    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
        )

        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence="c_v"
        )
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


data = datasets.load_dataset("amazon_polarity")["train"].select(range(10000))
text = []
for d in tqdm.tqdm(data):
    words = nltk.word_tokenize(d["title"]) + nltk.word_tokenize(d["content"])
    words = nltk.pos_tag(words)
    text.append([x[0].lower() for x in words if "NN" in x[1]])

dictionary = Dictionary(text)
corpus = [dictionary.doc2bow(doc) for doc in text]

model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    num_topics=20,
    id2word=dictionary,
)

model.print_topics()
