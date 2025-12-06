import torch
import torch.nn as nn

import numpy as np

from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer

from transformers import BertModel, BertConfig

from data import TokenData

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")


class BERTWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, x):
        return self.model(**x).pooler_output


class Transformer(nn.Module):
    def __init__(self, layers, embed, heads=8):
        super().__init__()

        vocab = BertConfig.from_pretrained("bert-base-uncased").vocab_size

        self.classToken = nn.Parameter(torch.zeros(1, 1, embed))
        self.position = nn.Parameter(torch.zeros(1, 512, embed))

        nn.init.normal_(self.classToken, mean=0, std=0.02)
        nn.init.normal_(self.position, mean=0, std=0.02)

        self.embeddings = nn.Embedding(vocab, embed)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed, dim_feedforward=embed * 4,
                nhead=heads, batch_first=True
            ),
            num_layers=layers
        )

    def forward(self, x):
        x = x["input_ids"]
        mask = (x == TokenData.tokenizer.pad_token_id)
        sequence = self.embeddings(x) + self.position[:, :x.shape[1]]
        # sequence = torch.cat([self.classToken.expand(x.shape[0], -1, -1), sequence], dim=1)
        outputs = self.encoder(sequence, src_key_padding_mask=mask)
        pooled = outputs[:, 0]
        return pooled


stemmer = PorterStemmer()
translator = str.maketrans('', '', "".join(string.punctuation))
stopWords = open("stopwords.txt").read().split("\n")
stopWords = set(stopWords)
analyzer = CountVectorizer().build_analyzer()


def stemmed(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


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

        print("Normalized Documents")

        vocab = set()
        for entry in normalized:
            abstract, citation = entry
            vocab = vocab.union(set(citation))

        vocab = list(vocab)
        vocabIndex = dict(zip(vocab, range(len(vocab))))

        print("Built Index")

        abstracts = [entry[0] for entry in data]
        citations = [entry[1] for entry in data]

        vectorizedAbstracts = CountVectorizer(max_features=len(vocab), analyzer=stemmed).fit_transform(abstracts)
        vectorizedCitations = CountVectorizer(max_features=len(vocab), analyzer=stemmed).fit_transform(citations)

        print(vectorizedAbstracts.shape, vectorizedAbstracts.dtype)

        print("Vectorized Documents")

        ids = np.array(ids)
        relevance = ids[:, None] == ids[None, :]

        self.documents = normalized
        self.vectorizedAbstracts = torch.tensor(vectorizedAbstracts.toarray(), dtype=torch.long)
        self.vectorizedCitations = torch.tensor(vectorizedCitations.toarray(), dtype=torch.long)
        self.relevance = relevance

        print("Ranking...")

    def rank(self):
        print(self.vectorizedCitations.shape)
        docTerm = self.vectorizedCitations > 0

        # Relevant = 1, xt = 1
        tl = torch.dot(self.relevance, docTerm)
        # Relevant = 0, xt = 1
        tr = torch.dot(1 - self.relevance, docTerm)

        # Relevant = 1, xt = 0
        bl = torch.dot(self.relevance, 1 - docTerm)
        # Relevant = 0, xt = 0
        br = torch.dot(1 - self.relevance, 1 - docTerm)

        w = torch.log(((tl + 0.5) / (bl + 0.5)) / ((tr + 0.5) / (br + 0.5)))

        queryTerm = self.vectorizedAbstracts > 0
        scores = torch.dot(docTerm, (w * queryTerm).t())

        return scores.cpu().numpy()
            

class BM25(BIM):
    def __init__(self, data: list[tuple[str, str]], ids: list[int]):
        super().__init__(data, ids)

    def rank(self, k1=1.5, b=0.75):
        docLengths = torch.tensor(np.array([len(doc) for doc in self.documents]), dtype=torch.long)
        avgDL = docLengths.mean()
        docTerm = self.vectorizedCitations > 0

        print(self.vectorizedCitations.shape)

        print(len(self.documents), np.sum(docTerm, axis=0))

        idf = torch.log10(len(self.documents) / torch.sum(docTerm, dim=0))
        numerator = self.vectorizedCitations * (k1 + 1)
        denominator = self.vectorizedCitations + (k1 * (1 - b + b * docLengths / avgDL))[:, None]

        queryTerm = self.vectorizedAbstracts > 0
        x = idf[:, None].t() * queryTerm
        scores = torch.dot(x, ((numerator / denominator) * queryTerm).t())
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
