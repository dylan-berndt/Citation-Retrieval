# Dylan Berndt and Faith Cordsiemon
# This script evaluates each of the information retrieval models on the Citation Network Dataset

from torch.utils.data import Dataset
from typing import List, Tuple

import numpy as np
import json
import os
import glob # Import glob for recursive file search
import ijson # Import ijson for streaming large JSON files
import sys # Import sys for printing debug info

from transformers import AutoTokenizer

import kagglehub


# Function for loading data from kagglehub
# Should return the plain text from each as well as an ID list lining up with the plain text
def loadData(max_data_size=None, subjects=None) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    Downloads the citation network dataset from KaggleHub, processes the JSON incrementally,
    and returns a list of (abstract, citation_title) pairs and corresponding paper IDs.
    
    NOTE: Uses ijson for low-memory streaming of the large JSON file.

    Args:
        max_data_size (int | None): If set to an integer, limits the total number of 
                                    (abstract, citation) pairs generated to this size.
        subjects (List[str]): Subjects to limit the results to

    Returns:
        tuple[list[tuple[str, str]], list[int]]: 
            - A list of tuples, where each tuple is (abstract_text, cited_paper_title).
            - A list of integers representing the ID of the abstracting paper.
    """
    # 1. Download the dataset
    dataset_root_path = kagglehub.dataset_download("mathurinache/citation-network-dataset")
    print(f"Dataset downloaded to: {dataset_root_path}")
    
    # 2. Locate the JSON file using a recursive search
    json_filename = "dblp.v12.json"
    
    try:
        json_path = next(iter(glob.iglob(os.path.join(dataset_root_path, '**', json_filename), recursive=True)))
        print(f"Found JSON file at: {json_path}")
    except StopIteration:
        json_filename = "Citation Network.json"
        try:
            json_path = next(iter(glob.iglob(os.path.join(dataset_root_path, '**', json_filename), recursive=True)))
            print(f"Found JSON file at: {json_path}")
        except StopIteration:
            raise FileNotFoundError(
                f"Could not find the expected files ('dblp.v12.json' or 'Citation Network.json') recursively within "
                f"the downloaded dataset path: {dataset_root_path}"
            )

    # 3. First Pass: Collect all paper IDs and their titles/references
    paper_map = {}
    print("Starting first pass to build paper map (IDs, Titles, References, Abstracts)...")
    
    # Debug flag to print the structure of the first failed indexed_abstract
    # debug_printed = False # Removed debugging flag

    if subjects is None:
        subjects = [""]

    with open(json_path, 'rb') as f: # Use 'rb' mode for ijson
        papers_stream = ijson.items(f, 'item')

        for i, raw_paper in enumerate(papers_stream):
            paper_id = raw_paper.get('id')
            if paper_id is None:
                continue

            if subjects[0] != "":
                if raw_paper.get("fos") is None:
                    continue
                paper_subjects = [subject["name"] for subject in raw_paper.get("fos")]
                intersection = set(subjects).intersection(paper_subjects)
                if len(intersection) != len(subjects):
                    print(f"\r{i} papers loaded, {len(paper_map)} usable", end="")
                    continue

            # --- Robust Abstract Extraction Logic ---
            abstract = ""

            # 1. Try to get abstract from the top-level 'abstract' field (simple string)
            if isinstance(raw_paper.get('abstract'), str):
                abstract = raw_paper['abstract']

            # 2. If not found, check the 'indexed_abstract' structure
            elif 'indexed_abstract' in raw_paper:
                indexed_abstract = raw_paper['indexed_abstract']
                temp_abstract = ""

                # Case 2a: Abstract text is a list of words/tokens under 'abstract'
                abstract_tokens = indexed_abstract.get('abstract')
                if isinstance(abstract_tokens, list) and abstract_tokens:
                    temp_abstract = " ".join(abstract_tokens)

                # Case 2b: Abstract is stored as an inverted index (word -> [positions])
                # This handles the structure identified in your debug output.
                inverted_index = indexed_abstract.get('InvertedIndex')
                index_length = indexed_abstract.get('IndexLength')

                if not temp_abstract and isinstance(inverted_index, dict) and isinstance(index_length, int) and index_length > 0:
                    reconstructed_tokens = [""] * index_length
                    all_indices_found = True

                    for word, indices in inverted_index.items():
                        if isinstance(indices, list):
                            for index in indices:
                                if 0 <= index < index_length:
                                    reconstructed_tokens[index] = word
                                else:
                                    # Index out of bounds, data might be messy. Flag and break.
                                    all_indices_found = False
                                    break
                        if not all_indices_found:
                            break

                    # Check if every position was filled, otherwise the abstract is incomplete/corrupt
                    # Note: We rely on the InvertedIndex being complete to reconstruct the abstract.
                    if all_indices_found and all(token for token in reconstructed_tokens):
                        temp_abstract = " ".join(reconstructed_tokens)

                # Case 2c (Original 2b): Abstract is stored as a dictionary of indexed segments, often under 'index'
                # The values of the dictionary are lists of tokens, keyed by position string ("0", "1", etc.)
                elif not temp_abstract and isinstance(indexed_abstract, dict) and 'index' in indexed_abstract and isinstance(indexed_abstract['index'], dict):
                    token_dict = indexed_abstract['index']
                    tokens = []

                    try:
                        # Attempt to sort by integer key (for "0", "1", "2", ...)
                        # Fall back to string sort if integer conversion fails for a key.
                        sorted_keys = sorted(token_dict.keys(), key=lambda x: int(x) if x.isdigit() else x)
                    except:
                        # If numerical sorting fails due to inconsistent keys, fall back to string sort
                        sorted_keys = sorted(token_dict.keys())

                    for key in sorted_keys:
                        if isinstance(token_dict.get(key), list):
                            tokens.extend(token_dict[key])

                    if tokens:
                        temp_abstract = " ".join(tokens)

                # Case 2d (Original 2c - Fallback): Look for the first list of strings with significant length
                if not temp_abstract and isinstance(indexed_abstract, dict):
                    for value in indexed_abstract.values():
                        if isinstance(value, list) and all(isinstance(t, str) for t in value) and len(value) > 10:
                            # Found a substantial list of strings, assume it's the abstract tokens
                            temp_abstract = " ".join(value)
                            break

                abstract = temp_abstract

            # Ensure abstract is not just empty or whitespace
            if abstract and abstract.strip():
                paper_map[paper_id] = {
                    'title': raw_paper.get('title', ''),
                    'references': raw_paper.get('references', []),
                    'abstract': abstract # Store the extracted abstract string
                }

            print(f"\r{i} papers loaded, {len(paper_map)} usable", end="")

            if max_data_size is not None and len(paper_map) > max_data_size:
                break

        print()
    
    print(f"First pass complete. Total unique papers with abstract and ID mapped: {len(paper_map)}")
    
    # 4. Second Pass (In-memory map processing): Process the data to create (abstract, citation_title) pairs
    abstract_citation_pairs = []
    abstract_paper_ids = []

    print("Starting second pass to generate (abstract, citation) pairs...")
    
    for abstracting_paper_id, paper_data in paper_map.items():
        # Stop processing if the maximum size limit is reached
        if max_data_size is not None and len(abstract_citation_pairs) >= max_data_size:
            break
            
        # Abstract is now directly available as a string
        abstract = paper_data['abstract']
        references = paper_data.get('references')
        
        # We already filtered papers with abstracts in the first pass.
        # Check if the paper has references
        if references:
            
            # Find the full paper objects for all referenced IDs
            for referenced_id in references:
                # Check limit again inside the inner loop, as one paper can generate multiple pairs
                if max_data_size is not None and len(abstract_citation_pairs) >= max_data_size:
                    break
                    
                referenced_paper_data = paper_map.get(referenced_id)
                
                # Check if the referenced paper exists in our map and has a title
                if referenced_paper_data and referenced_paper_data.get('title'):
                    citation_title = referenced_paper_data['title']
                    
                    # Store the (abstract, citation_title) pair and the ID of the paper with the abstract
                    abstract_citation_pairs.append((abstract, citation_title))
                    abstract_paper_ids.append(abstracting_paper_id)
    
    # Truncate to max_data_size if necessary (this handles cases where the last paper pushed it over the limit)
    if max_data_size is not None:
        abstract_citation_pairs = abstract_citation_pairs[:max_data_size]
        abstract_paper_ids = abstract_paper_ids[:max_data_size]


    print(f"Data loading complete. Total {len(abstract_citation_pairs)} (abstract, citation) pairs generated.")
    return abstract_citation_pairs, abstract_paper_ids


# data: List of tuples including each abstract -> citation pair
# There should be duplicate abstracts in the list since each should have multiple citations
class TokenData(Dataset):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def __init__(self, data: list[tuple[str, str]], ids: list[int]):
        # ids is not strictly needed for TokenData but kept for compatibility with BIM/BM25
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return {"abstract": self.data[i][0], "citation": self.data[i][1]}

    @staticmethod
    def collate(samples):
        abstracts = [sample["abstract"].lower() for sample in samples]
        citations = [sample["citation"].lower() for sample in samples]

        # Tokenize both abstracts (queries) and citations (documents)
        abstracts = TokenData.tokenizer(
            abstracts, 
            padding="longest",
            truncation=True, # Added truncation for safety
            max_length=256,  # Standard BERT max length
            return_tensors="pt"
        )
        citations = TokenData.tokenizer(
            citations, 
            padding="longest",
            truncation=True, # Added truncation for safety
            max_length=256,  # Standard BERT max length
            return_tensors="pt"
        )

        return abstracts, citations