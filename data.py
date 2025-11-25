from torch.utils.data import Dataset

import numpy as np

from transformers import AutoTokenizer

import kagglehub


# Function for loading data from kagglehub
# Should return the plain text from each as well as an ID list lining up with the plain text
def loadData() -> tuple[list[tuple[str, str]], list[int]]:
    pass


# data: List of tuples including each abstract -> citation pair
# There should be duplicate abstracts in the list since each should have multiple citations
class TokenData(Dataset):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def __init__(self, data: list[tuple[str, str]], ids: list[int]):
        self.data = np.array(data)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, i):
        return {"abstract": self.data[i][0], "citation": self.data[i][1]}

    @staticmethod
    def collate(samples):
        abstracts = [sample["abstract"] for sample in samples]
        citations = [sample["citation"] for sample in samples]

        abstracts = TokenData.tokenizer(abstracts, padding="max_length", return_tokens="pt")
        citations = TokenData.tokenizer(citations, padding="max_length", return_tokens="pt")

        return abstracts, citations