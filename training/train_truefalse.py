import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from models.transformer import TransformerClassifier, TransformerModel
from utils.tokenizer import tokenizer
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

class TFDataset(Dataset):
    def __init__(self, texts, labels, max_len=128, max_model_length=1024):
        self.max_len = min(max_len, max_model_length)
        self.max_model_length = max_model_length
        
        print(f"Processing {len(texts)} samples with max_len={self.max_len}")
        
        # Tokenize and truncate texts
        self.inputs = []
        self.labels = []
        
        for i, (text, label) in enumerate(zip(texts, labels)):
            try:
                # Tokenize text
                token_ids = tokenizer.encode(text)
                
                # Truncate if too long
                if len(token_ids) > self.max_len:
                    token_ids = token_ids[:self.max_len]
                
                # Skip if too short (less than 5 tokens)
                if len(token_ids) < 5:
                    print(f"Skipping sample {i}: too short ({len(token_ids)} tokens)")
                    continue
                
                self.inputs.append(torch.tensor(token_ids, dtype=torch.long))
                self.labels.append(label)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        print(f"Successfully processed {len(self.inputs)} samples")
        
        if len(self.inputs) == 0:
            raise ValueError("No valid samples found after processing")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    inputs, labels = zip(*batch)
    
    # Pad sequences to the same length
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    
    return inputs, labels

def load_tf_dataset(path):
    statements, labels = [], []
    
    print(f"Loading dataset from {path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Try to split by tab
                    if '\t' in line:
                        parts = line.rsplit('\t', 1)
                    else:
                        # Fallback to space split
                        parts = line.rsplit(' ', 1)
                    
                    if len(parts) != 2:
                        print(f"Warning: Line {line_num} doesn't have exactly 2 parts: {line}")
                        continue
                    
                    statement, label_str = parts
                    label = int(label_str.strip())
                    
                    if label not in [0, 1]:
                        print(f"Warning: Line {line_num} has invalid label {label}, skipping")
                        continue
                    
                    statements.append(statement.strip())
                    labels.append(label)
                    
                except ValueError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    print(f"Loaded {len(statements)} statements")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return statements, labels

def setup_device():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA available: {num_gpus} GPU(s) detected")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return 'cuda', num_gpus
    else:
        print("CUDA not available, using CPU")
        return 'cpu', 0

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()

def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    return avg_loss, accuracy, precision, recall, f1

def train_single_gpu(statements, labels, epochs=250, batch_size=8, max_len=128, lr=1e-4, 
                    device='cuda', max_model_length=1024, hidden_dim=256):
    print(f"Training on {device}")
    
    # Create dataset
    dataset = TFDataset(statements, labels, max_len=max_len, max_model_length=max_model_length)
    
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=collate_fn, num_workers=4)
    
    # Initialize model
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    base_model = TransformerModel(vocab_size=vocab_size, max_seq_len=max_len)
    model = TransformerClassifier(base_model, hidden_dim=hidden_dim)
    model = model.to(device)
    model.train()
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          factor=0.5, patience=5)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Move to device
            x, y = x.to(device), y.to(device)
            
            # Debug print for first batch
            if epoch == 0 and batch_idx == 0:
                print(f"First batch - Input shape: {x.shape}, Target shape: {y.shape}")
                print(f"Max token in batch: {x.max().item()}")
                print(f"Min token in batch: {x.min().item()}")
                print(f"Label distribution in batch: {torch.bincount(y)}")
            
            optimizer.zero_grad()
            
            try:
                logits = model(x)
                loss = criterion(logits, y)
                
            except Exception as e:
                print(f"Model forward pass failed:")
                print(f"  Input shape: {x.shape}")
                print(f"  Input device: {x.device}")
                print(f"  Error: {e}")
                raise e
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            num_batches += 1
            
            # Print progress every 20 batches
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(
            model, val_loader, device, criterion)
        
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}, "
              f"Val F1 = {val_f1:.4f}")
        
        # Early stopping
        if epoch > 20 and val_loss > min(val_losses[-10:]) * 1.1:
            print("Early stopping triggered")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    final_val_loss, final_accuracy, final_precision, final_recall, final_f1 = evaluate_model(
        model, val_loader, device, criterion)
    
    print(f"\nFinal Results:")
    print(f"  Validation Loss: {final_val_loss:.4f}")
    print(f"  Validation Accuracy: {final_accuracy:.4f}")
    print(f"  Validation Precision: {final_precision:.4f}")
    print(f"  Validation Recall: {final_recall:.4f}")
    print(f"  Validation F1: {final_f1:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/tf_classifier.pth")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Learning rate plot
    plt.subplot(1, 3, 3)
    lr_history = [group['lr'] for group in optimizer.param_groups]
    plt.plot(range(1, len(train_losses)+1), [lr_history[0]] * len(train_losses))
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("models/tf_classifier_training_curves.png")
    plt.show()
    
    return train_losses, val_losses, val_accuracies

def train_distributed_gpu(rank, world_size, statements, labels, epochs=250, batch_size=8, 
                         max_len=128, lr=1e-4, max_model_length=1024, hidden_dim=256):
    print(f"Starting distributed training on rank {rank}")
    
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Create dataset
    dataset = TFDataset(statements, labels, max_len=max_len, max_model_length=max_model_length)
    
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, 
                             collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, 
                           collate_fn=collate_fn, num_workers=4)
    
    # Initialize model
    vocab_size = tokenizer.vocab_size
    base_model = TransformerModel(vocab_size=vocab_size, max_seq_len=max_len)
    model = TransformerClassifier(base_model, hidden_dim=hidden_dim)
    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          factor=0.5, patience=5)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  # Important for proper shuffling
        
        # Training phase
        model.train()
        total_train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(rank), y.to(rank)
            
            optimizer.zero_grad()
            
            logits = model(x)
            loss = criterion(logits, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            num_batches += 1
            
            # Print progress only on rank 0
            if rank == 0 and batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss, val_accuracy, _, _, val_f1 = evaluate_model(model, val_loader, rank, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        scheduler.step(val_loss)
        
        if rank == 0:
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}, "
                  f"Val F1 = {val_f1:.4f}")
    
    # Save model only on rank 0
    if rank == 0:
        os.makedirs("models", exist_ok=True)
        torch.save(model.module.state_dict(), "models/tf_classifier.pth")
        
        # Plot training curves
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss (Distributed)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy (Distributed)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("models/tf_classifier_training_curves.png")
        plt.show()
    
    cleanup_distributed()
    return train_losses, val_losses, val_accuracies

def train_true_false_classifier(statements, labels, epochs=250, batch_size=8, max_len=128, 
                               lr=1e-4, force_cpu=False, num_gpus=None, max_model_length=1024, 
                               hidden_dim=256):
    print(f"Starting true/false classifier training at {datetime.now()}")
    print(f"Parameters:")
    print(f"  Number of samples: {len(statements)}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max sequence length: {max_len}")
    print(f"  Learning rate: {lr}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Max model length: {max_model_length}")
    
    # Setup device
    device_type, available_gpus = setup_device()
    
    if force_cpu or device_type == 'cpu':
        # Train on CPU
        losses = train_single_gpu(statements, labels, epochs, batch_size, max_len, lr, 'cpu',
                                max_model_length, hidden_dim)
    elif available_gpus == 1 or num_gpus == 1:
        # Train on single GPU
        losses = train_single_gpu(statements, labels, epochs, batch_size, max_len, lr, 'cuda',
                                max_model_length, hidden_dim)
    else:
        # Train on multiple GPUs
        world_size = num_gpus if num_gpus is not None else available_gpus
        world_size = min(world_size, available_gpus)
        
        print(f"Starting distributed training on {world_size} GPUs")
        
        # Use multiprocessing to spawn processes for each GPU
        mp.spawn(train_distributed_gpu, 
                args=(world_size, statements, labels, epochs, batch_size, max_len, lr, max_model_length, hidden_dim),
                nprocs=world_size, 
                join=True)
        
        losses = []  # Losses are handled in distributed function
    
    print(f"Training completed at {datetime.now()}")
    return losses

def main():
    parser = argparse.ArgumentParser(description='Train True/False Classifier')
    parser.add_argument('--dataset_path', type=str, default="data/tf_statements_dataset.txt",
                        help='Path to training dataset file')
    parser.add_argument('--epochs', type=int, default=250, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--num_gpus', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--max_model_length', type=int, default=1024, 
                        help='Maximum sequence length the model can handle')
    parser.add_argument('--hidden_dim', type=int, default=256, 
                        help='Hidden dimension for classifier')
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    try:
        # Load dataset
        statements, labels = load_tf_dataset(args.dataset_path)
        
        # Train model
        train_true_false_classifier(
            statements=statements,
            labels=labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_len=args.max_len,
            lr=args.lr,
            force_cpu=args.force_cpu,
            num_gpus=args.num_gpus,
            max_model_length=args.max_model_length,
            hidden_dim=args.hidden_dim
        )
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()