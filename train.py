# Dylan Berndt and Faith Cordsiemon
# This script evaluates each of the information retrieval models on the Citation Network Dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import os

from data import loadData, TokenData
from models import BERTWrapper, Transformer
from modules import InfoNCE # Assuming InfoNCE is correctly implemented in modules.py

DataSize = None
Subjects = ["Machine learning"]


def iterate(dataset):
    while True:
        for data in dataset:
            yield data


def train_model(
    model: nn.Module,
    train: DataLoader,
    test: DataLoader,
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer,
    device: torch.device
):
    """
    Performs one epoch of training on the provided model.
    """
    total_loss = 0
    total_test_loss = 0

    test_iterator = iterate(test)
    
    for batch_idx, (abstracts_tokens, citations_tokens) in enumerate(train):
        # Move tokenized inputs to the specified device
        abstracts_tokens = {k: v.to(device) for k, v in abstracts_tokens.items()}
        citations_tokens = {k: v.to(device) for k, v in citations_tokens.items()}

        model.train()
        optimizer.zero_grad()

        # 1. Generate embeddings (y1 for abstract, y2 for citation)
        # NOTE: BERTWrapper expects the input_ids tensor directly, but transformers tokenizer 
        # returns a dictionary. We pass the input_ids.
        y1 = model(abstracts_tokens)
        y2 = model(citations_tokens)

        # 2. Calculate the InfoNCE loss
        # InfoNCE loss computes the similarity matrix (logits) and applies CrossEntropyLoss
        # where the positive samples are on the diagonal (abstract_i -> citation_i).
        loss = criterion(y1, y2)
        
        # 3. Backpropagation and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        model.eval()
        test_abstract_tokens, test_citation_tokens = next(test_iterator)
        test_abstract_tokens = {k: v.to(device) for k, v in test_abstract_tokens.items()}
        test_citation_tokens = {k: v.to(device) for k, v in test_citation_tokens.items()}
        y1 = model(test_abstract_tokens)
        y2 = model(test_citation_tokens)
        test_loss = criterion(y1, y2)

        total_test_loss += test_loss.item()

        print(f"\r{batch_idx + 1}/{len(train)} | Train Loss: {loss.item():.2f} | Test Loss: {test_loss.item():.2f}", end="")

    print()

    avg_loss = total_loss / len(train)
    avg_test_loss = total_test_loss / len(train)
    print(f"Epoch finished. Average Train Loss: {avg_loss:.4f} | Average Test Loss {avg_test_loss:.4f}")
    return avg_loss


def main():
    # --- Configuration ---
    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 20
    
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.environ["KAGGLEHUB_CACHE"] = "F:/.cache/kagglehub"

    # --- Data Loading ---
    print("Loading data...")
    data_pairs, paper_ids = loadData(DataSize, subjects=Subjects)
    
    # A simple split for demonstration; in production, use dedicated train/val/test splits
    # Since the full dataset is small, we'll use a subset to quickly verify the training loop
    # For a real project, shuffle and split data_pairs and paper_ids consistently.
    split_idx = int(0.8 * len(data_pairs))
    train_data = data_pairs[:split_idx]
    test_data = data_pairs[split_idx:]
    
    train_dataset = TokenData(train_data, paper_ids[:split_idx])
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=TokenData.collate
    )

    test_dataset = TokenData(test_data, paper_ids[split_idx:])
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE // 4,
        shuffle=True,
        collate_fn=TokenData.collate
    )

    # --- Model Setup ---
    # query_encoder = BERTWrapper().to(device) # Your BERT model wrapper
    # text_encoder = BERTWrapper().to(device)

    model = Transformer(8, 256, heads=8).to(device)
    
    # --- Loss and Optimizer Setup ---
    # InfoNCE is typically used for contrastive learning in retrieval tasks
    criterion = InfoNCE(temperature=0.03).to(device) # Common temperature value for InfoNCE
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("Starting training...")
    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
            train_model(model, train_dataloader, test_dataloader, criterion, optimizer, device)
    except KeyboardInterrupt:
        pass
    
    # --- Save Model ---
    output_path = "citation_retrieval_model.pth"
    torch.save(model.state_dict(), output_path)
    print(f"\nTraining complete. Model saved to {output_path}")


if __name__ == "__main__":
    # Ensure all imports from data.py, models.py, modules.py are correct
    # and all dependencies (kagglehub, transformers, torch) are installed.
    main()