import torch
import torch.nn as nn

import numpy as np

from nltk.stem import PorterStemmer
import string

from transformers import BertModel


class BERTWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, x):
        return self.model(x).pooler_output


stemmer = PorterStemmer()
translator = str.maketrans('', '', "".join(string.punctuation))
stopWords = open("stopwords.txt").read().split("\n")
stopWords = set(stopWords)


def normalizeDoc(document):
    tokenized = document.split()
    doc = set([word.lower().translate(translator) for word in tokenized])
    stopped = list(doc - stopWords)
    normalized = [stemmer.stem(word) for word in stopped]

    return normalized


def vectorizeDocument(document, vocabIndex):
    vector = np.zeros([len(vocabIndex)])
    for word in document:
        if word not in vocabIndex:
            continue
        num = vocabIndex[word]
        vector[num] += 1
    return vector


class BIM:
    def __init__(self, data: list[tuple[str, str]], ids: list[int]):
        normalized = [(normalizeDoc(entry[0]), normalizeDoc(entry[1])) for entry in data]

        vocab = set()
        for entry in normalized:
            abstract, citation = entry
            vocab = vocab.union(set(citation))

        vocab = list(vocab)
        vocabIndex = dict(zip(vocab, range(len(vocab))))

        vectorizedAbstracts = [vectorizeDocument(entry[0], vocabIndex) for entry in normalized]
        vectorizedCitations = [vectorizeDocument(entry[1], vocabIndex) for entry in normalized]

        ids = np.array(ids)
        relevance = ids[:, None] == ids[None, :]

        self.documents = normalized
        self.vectorizedAbstracts = np.array(vectorizedAbstracts)
        self.vectorizedCitations = np.array(vectorizedCitations)
        self.relevance = relevance

    def rank(self):
        docTerm = self.vectorizedCitations > 0

        # Relevant = 1, xt = 1
        tl = np.dot(self.relevance, docTerm)
        # Relevant = 0, xt = 1
        tr = np.dot(1 - self.relevance, docTerm)

        # Relevant = 1, xt = 0
        bl = np.dot(self.relevance, 1 - docTerm)
        # Relevant = 0, xt = 0
        br = np.dot(1 - self.relevance, 1 - docTerm)

        w = np.log(((tl + 0.5) / (bl + 0.5)) / ((tr + 0.5) / (br + 0.5)))

        queryTerm = self.vectorizedAbstracts > 0
        scores = np.dot(docTerm, (w * queryTerm).T)

        return scores
            

class BM25(BIM):
    def __init__(self, data: list[tuple[str, str]], ids: list[int]):
        super().__init__(data, ids)

    def rank(self, k1=1.5, b=0.75):
        docLengths = np.array([len(doc) for doc in self.documents])
        avgDL = docLengths.mean()
        docTerm = self.vectorizedCitations > 0

        print(self.vectorizedCitations.shape)

        print(len(self.documents), np.sum(docTerm, axis=0))

        idf = np.log10(len(self.documents) / np.sum(docTerm, axis=0))
        numerator = self.vectorizedCitations * (k1 + 1)
        denominator = self.vectorizedCitations + (k1 * (1 - b + b * docLengths / avgDL))[:, None]

        queryTerm = self.vectorizedAbstracts > 0
        x = idf[:, None].T * queryTerm
        scores = np.dot(x, ((numerator / denominator) * queryTerm).T)
        return scores


if __name__ == "__main__":
    data = [
        ("Hello, I am a query", "Wow, that's interesting, I'm a response"),
        ("Hey there", "What's up"),
        ("Hello, I am a query", "Interesting introduction big guy"),
        ("Hey there", "Hello, I am your guide for today"),
        ("What's up folks", "Hey, how's it going?")
    ]
    ids = [0, 1, 0, 1, 2]

    scores = BIM(data, ids).rank()
    print(scores)

    scores = BM25(data, ids).rank()
    print(scores)