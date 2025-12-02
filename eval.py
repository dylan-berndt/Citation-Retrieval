import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from data import loadData, TokenData
from models import BERTWrapper
from modules import MAP # Assuming MAP is correctly implemented in modules.py


def get_embeddings(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """
    Generates embeddings for all abstracts (queries) and citations (documents).
    """
    model.eval()
    all_abstract_embeddings = []
    all_citation_embeddings = []

    with torch.no_grad():
        for abstracts_tokens, citations_tokens in tqdm(dataloader, desc="Generating Embeddings"):
            abstracts_tokens = {k: v.to(device) for k, v in abstracts_tokens.items()}
            citations_tokens = {k: v.to(device) for k, v in citations_tokens.items()}

            y1 = model(abstracts_tokens['input_ids']) # Abstract embeddings (Queries)
            y2 = model(citations_tokens['input_ids']) # Citation embeddings (Documents)

            all_abstract_embeddings.append(y1.cpu().numpy())
            all_citation_embeddings.append(y2.cpu().numpy())

    return np.concatenate(all_abstract_embeddings), np.concatenate(all_citation_embeddings)


def calculate_map(embeddings_q, embeddings_d, paper_ids):
    """
    Calculates the relevance scores and Mean Average Precision (MAP).

    Args:
        embeddings_q (np.ndarray): Embeddings for abstracts (Queries).
        embeddings_d (np.ndarray): Embeddings for citations (Documents).
        paper_ids (list): List of IDs corresponding to the abstracts/citations.
    """
    print("Calculating relevance matrix...")
    
    # 1. Calculate similarity scores (dot product) between all queries and all documents
    # The result is a (N_queries x N_documents) matrix
    # Normalize embeddings before calculating similarity for better results
    embeddings_q = embeddings_q / np.linalg.norm(embeddings_q, axis=1, keepdims=True)
    embeddings_d = embeddings_d / np.linalg.norm(embeddings_d, axis=1, keepdims=True)
    
    # Compute the relevance/similarity matrix
    similarity_scores = embeddings_q @ embeddings_d.T
    
    # 2. Create the true relevance matrix
    # This matrix tells us if Query_i is a positive match for Document_j.
    # A query is relevant to a document if they share the same source paper ID.
    ids_array = np.array(paper_ids)
    true_relevance = (ids_array[:, None] == ids_array[None, :]).astype(int)
    
    # 3. Calculate MAP
    # The MAP module will use the similarity scores as predictions and the true_relevance matrix as targets.
    # NOTE: The provided modules.py has a placeholder for MAP, so we assume a function for MAP calculation here.
    # Since MAP is typically a classification or ranking metric, let's implement a simplified MAP logic here.

    # Find the top 100 ranks for each query
    # Get the indices that would sort the scores in descending order
    sorted_indices = np.argsort(-similarity_scores, axis=1)

    # Initialize MAP calculation variables
    num_queries = similarity_scores.shape[0]
    average_precisions = []
    
    # Iterate over each query
    for i in tqdm(range(num_queries), desc="Calculating MAP"):
        # True relevance vector for the i-th query (0 or 1 for each document)
        y_true = true_relevance[i, :]
        # Ranked document indices for the i-th query (by predicted similarity)
        ranked_doc_indices = sorted_indices[i, :]
        
        # For retrieval, we only care about the documents that are positive matches
        relevant_documents = np.where(y_true == 1)[0]
        
        if len(relevant_documents) == 0:
            average_precisions.append(0.0)
            continue

        # Keep track of relevant documents found so far
        num_relevant_found = 0
        precision_at_k_sum = 0
        
        # Iterate through the ranked list
        for k in range(len(ranked_doc_indices)):
            doc_idx = ranked_doc_indices[k]
            
            # Check if the ranked document is truly relevant
            if y_true[doc_idx] == 1:
                num_relevant_found += 1
                
                # Precision at k: (Relevant items found up to k) / (Total items retrieved up to k)
                precision_at_k = num_relevant_found / (k + 1)
                precision_at_k_sum += precision_at_k
        
        # Average Precision (AP) for this query is the mean of P@k for each relevant document found
        AP = precision_at_k_sum / len(relevant_documents)
        average_precisions.append(AP)

    # Mean Average Precision (MAP) is the mean of APs across all queries
    MAP_score = np.mean(average_precisions)
    return MAP_score


def main():
    # --- Configuration ---
    BATCH_SIZE = 32
    MODEL_PATH = "citation_retrieval_model.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading data...")
    data_pairs, paper_ids = loadData()
    
    # Use the full dataset for evaluation
    eval_dataset = TokenData(data_pairs, paper_ids)
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, # No need to shuffle for evaluation
        collate_fn=TokenData.collate
    )

    # --- Model Setup ---
    model = BERTWrapper().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model weights loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}. Please run train.py first.")
        return

    # --- Evaluation ---
    abstract_embeddings, citation_embeddings = get_embeddings(model, eval_dataloader, device)
    
    # Calculate MAP using the generated embeddings and paper IDs
    map_score = calculate_map(abstract_embeddings, citation_embeddings, paper_ids)
    
    print(f"\nEvaluation Complete.")
    print(f"Mean Average Precision (MAP) Score: {map_score:.4f}")


if __name__ == "__main__":
    main()