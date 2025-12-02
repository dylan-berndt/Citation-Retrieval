import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from data import loadData, TokenData
from models import BERTWrapper
from modules import InfoNCE # Assuming InfoNCE is correctly implemented in modules.py

DataSize = 1000

def train_model(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device
):
    """
    Performs one epoch of training on the provided model.
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (abstracts_tokens, citations_tokens) in enumerate(tqdm(dataloader, desc="Training")):
        # Move tokenized inputs to the specified device
        abstracts_tokens = {k: v.to(device) for k, v in abstracts_tokens.items()}
        citations_tokens = {k: v.to(device) for k, v in citations_tokens.items()}
        
        optimizer.zero_grad()

        # 1. Generate embeddings (y1 for abstract, y2 for citation)
        # NOTE: BERTWrapper expects the input_ids tensor directly, but transformers tokenizer 
        # returns a dictionary. We pass the input_ids.
        y1 = model(abstracts_tokens['input_ids'])
        y2 = model(citations_tokens['input_ids'])

        # 2. Calculate the InfoNCE loss
        # InfoNCE loss computes the similarity matrix (logits) and applies CrossEntropyLoss
        # where the positive samples are on the diagonal (abstract_i -> citation_i).
        loss = criterion(y1, y2)
        
        # 3. Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch finished. Average Loss: {avg_loss:.4f}")
    return avg_loss


def main():
    # --- Configuration ---
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading data...")
    data_pairs, paper_ids = loadData(DataSize)
    
    # A simple split for demonstration; in production, use dedicated train/val/test splits
    # Since the full dataset is small, we'll use a subset to quickly verify the training loop
    # For a real project, shuffle and split data_pairs and paper_ids consistently.
    split_idx = int(0.9 * len(data_pairs))
    train_data = data_pairs[:split_idx]
    
    train_dataset = TokenData(train_data, paper_ids[:split_idx])
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=TokenData.collate
    )

    # --- Model Setup ---
    model = BERTWrapper().to(device) # Your BERT model wrapper
    
    # --- Loss and Optimizer Setup ---
    # InfoNCE is typically used for contrastive learning in retrieval tasks
    criterion = InfoNCE(temperature=0.07).to(device) # Common temperature value for InfoNCE
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        train_model(model, train_dataloader, criterion, optimizer, device)
    
    # --- Save Model ---
    output_path = "citation_retrieval_model.pth"
    torch.save(model.state_dict(), output_path)
    print(f"\nTraining complete. Model saved to {output_path}")


if __name__ == "__main__":
    # Ensure all imports from data.py, models.py, modules.py are correct
    # and all dependencies (kagglehub, transformers, torch) are installed.
    main()