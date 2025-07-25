import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from models.transformer import TransformerModel
from utils.tokenizer import tokenizer
from utils.text_loader import load_text_file
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

class QuestionDataset(Dataset):
    def __init__(self, path, seq_len=128, max_model_length=1024, stride=None):
        self.seq_len = min(seq_len, max_model_length)
        self.stride = stride if stride is not None else self.seq_len
        self.max_model_length = max_model_length
        
        text = load_text_file(path)
        
        # Split by lines (one question per line) and filter short lines
        questions = [line.strip() for line in text.split("\n") if len(line.strip()) > 10]
        print(f"Found {len(questions)} questions")
        
        # Join questions with special separator or newline
        joined = " ".join(questions)
        token_ids = tokenizer.encode(joined)
        
        print(f"Original text tokenized to {len(token_ids)} tokens")
        
        # Validate sequence length
        if self.seq_len > max_model_length:
            print(f"Warning: seq_len {seq_len} exceeds max_model_length {max_model_length}")
            print(f"Setting seq_len to {max_model_length}")
            self.seq_len = max_model_length
        
        # Create samples with sliding window approach
        self.samples = []
        if len(token_ids) <= self.seq_len:
            # If text is shorter than sequence length, pad it
            if len(token_ids) < self.seq_len:
                padded_tokens = token_ids + [token_ids[-1]] * (self.seq_len - len(token_ids))
                self.samples.append(torch.tensor(padded_tokens, dtype=torch.long))
            else:
                self.samples.append(torch.tensor(token_ids, dtype=torch.long))
        else:
            # Create overlapping sequences
            for i in range(0, len(token_ids) - self.seq_len + 1, self.stride):
                sequence = token_ids[i:i + self.seq_len]
                if len(sequence) == self.seq_len:
                    self.samples.append(torch.tensor(sequence, dtype=torch.long))
        
        print(f"Created {len(self.samples)} samples from {len(token_ids)} tokens")
        print(f"Sequence length: {self.seq_len}, Stride: {self.stride}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence = self.samples[idx]
        
        # Ensure sequence doesn't exceed max model length
        if len(sequence) > self.max_model_length:
            sequence = sequence[:self.max_model_length]
        
        x = sequence[:-1]  # Input: all tokens except last
        y = sequence[1:]   # Target: all tokens except first
        
        assert len(x) <= self.max_model_length - 1, f"Input sequence length {len(x)} exceeds max {self.max_model_length - 1}"
        assert len(y) <= self.max_model_length - 1, f"Target sequence length {len(y)} exceeds max {self.max_model_length - 1}"
        
        return x, y

def check_model_config(model, max_seq_len=1024):
    """Check if model supports the requested sequence length"""
    print(f"Checking model configuration for max sequence length: {max_seq_len}")
    
    # Check if model has a config attribute
    if hasattr(model, 'config'):
        config = model.config
        if hasattr(config, 'max_position_embeddings'):
            max_pos = config.max_position_embeddings
            print(f"Model max_position_embeddings: {max_pos}")
            if max_pos < max_seq_len:
                print(f"WARNING: Model max_position_embeddings ({max_pos}) < requested max_seq_len ({max_seq_len})")
                return min(max_pos, max_seq_len)
    
    # Check for positional embeddings in the model
    for name, param in model.named_parameters():
        if 'pos_emb' in name or 'position_embedding' in name:
            pos_emb_size = param.shape[0] if param.dim() > 1 else param.shape[0]
            print(f"Found positional embedding '{name}' with size: {pos_emb_size}")
            if pos_emb_size < max_seq_len:
                print(f"WARNING: Positional embedding size ({pos_emb_size}) < requested max_seq_len ({max_seq_len})")
                return min(pos_emb_size, max_seq_len)
    
    print("Model configuration check passed")
    return max_seq_len

def setup_device():
    """Setup device configuration"""
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
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def train_single_gpu(txt_path, epochs=500, batch_size=8, seq_len=128, lr=1e-4, device='cuda',
                    max_model_length=1024, stride=None):
    """Train on single GPU or CPU"""
    print(f"Training on {device}")
    
    # Create dataset
    dataset = QuestionDataset(txt_path, seq_len=seq_len, max_model_length=max_model_length, stride=stride)
    
    if len(dataset) == 0:
        raise ValueError(f"No valid sequences created. Text too short or seq_len too large.")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize model
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Validate and adjust sequence length
    effective_seq_len = min(seq_len, max_model_length)
    if seq_len > max_model_length:
        print(f"Warning: Requested seq_len {seq_len} > max_model_length {max_model_length}")
        print(f"Using seq_len = {effective_seq_len}")
    
    model = TransformerModel(vocab_size=vocab_size, max_seq_len=effective_seq_len)
    
    # Check model configuration
    actual_max_length = check_model_config(model, max_model_length)
    if actual_max_length < max_model_length:
        print(f"Adjusting max_model_length from {max_model_length} to {actual_max_length}")
        max_model_length = actual_max_length
        
        # Recreate dataset with corrected max length
        if effective_seq_len > max_model_length:
            effective_seq_len = max_model_length
            dataset = QuestionDataset(txt_path, seq_len=effective_seq_len, 
                                    max_model_length=max_model_length, stride=stride)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = model.to(device)
    model.train()
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            # Validate input sequence length
            batch_size_actual, seq_len_actual = x.shape
            
            # Additional safety checks
            if seq_len_actual > max_model_length - 1:
                print(f"Error: Batch sequence length {seq_len_actual} exceeds max_model_length-1 {max_model_length-1}")
                x = x[:, :max_model_length-1]
                y = y[:, :max_model_length-1]
            
            # Move to device
            x, y = x.to(device), y.to(device)
            
            # Debug print for first batch
            if epoch == 0 and batch_idx == 0:
                print(f"First batch - Input shape: {x.shape}, Target shape: {y.shape}")
                print(f"Max token in batch: {x.max().item()}")
                print(f"Min token in batch: {x.min().item()}")
            
            optimizer.zero_grad()
            
            try:
                logits = model(x)
                # Ensure logits have the right shape
                if logits.dim() == 3:  # [batch_size, seq_len, vocab_size]
                    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")
                    
            except Exception as e:
                print(f"Model forward pass failed:")
                print(f"  Input shape: {x.shape}")
                print(f"  Input device: {x.device}")
                print(f"  Input dtype: {x.dtype}")
                print(f"  Error: {e}")
                raise e
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress every 50 batches
            #if batch_idx % 50 == 0:
                #print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Average loss = {avg_loss:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/question_gen_model.pth")
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Question Generation Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("models/question_gen_loss_plot.png")
    plt.show()
    
    return losses

def train_distributed_gpu(rank, world_size, txt_path, epochs=500, batch_size=8, seq_len=128, 
                         lr=1e-4, max_model_length=1024, stride=None):
    print(f"Starting distributed training on rank {rank}")
    
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Create dataset
    dataset = QuestionDataset(txt_path, seq_len=seq_len, max_model_length=max_model_length, stride=stride)
    
    if len(dataset) == 0:
        raise ValueError(f"No valid sequences created. Text too short or seq_len too large.")
    
    # Create distributed sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    
    # Initialize model
    vocab_size = tokenizer.vocab_size
    effective_seq_len = min(seq_len, max_model_length)
    
    model = TransformerModel(vocab_size=vocab_size, max_seq_len=effective_seq_len)
    
    # Check model configuration
    actual_max_length = check_model_config(model, max_model_length)
    if actual_max_length < max_model_length:
        max_model_length = actual_max_length
        if effective_seq_len > max_model_length:
            effective_seq_len = max_model_length
            # Note: In distributed training, you'd need to recreate dataset on all processes
    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            # Move to device
            x, y = x.to(rank), y.to(rank)
            
            optimizer.zero_grad()
            
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress only on rank 0
            #if rank == 0 and batch_idx % 50 == 0:
                #print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        if rank == 0:
            print(f"Epoch {epoch+1}: Average loss = {avg_loss:.4f}")
    
    # Save model only on rank 0
    if rank == 0:
        os.makedirs("models", exist_ok=True)
        torch.save(model.module.state_dict(), "models/question_gen_model.pth")
        
        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs+1), losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Question Generation Training Loss (Distributed)')
        plt.legend()
        plt.grid(True)
        plt.savefig("models/question_gen_loss_plot.png")
        plt.show()
    
    cleanup_distributed()
    return losses

def train_question_gen_model(txt_path, epochs=500, batch_size=8, seq_len=128, lr=1e-4,
                           force_cpu=False, num_gpus=None, max_model_length=1024, stride=None):
    print(f"Starting question generation training at {datetime.now()}")
    print(f"Parameters:")
    print(f"  Text path: {txt_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Learning rate: {lr}")
    print(f"  Max model length: {max_model_length}")
    print(f"  Stride: {stride}")
    
    # Validate parameters
    if seq_len > max_model_length:
        print(f"Warning: seq_len ({seq_len}) > max_model_length ({max_model_length})")
        print(f"Will use seq_len = {max_model_length}")
    
    # Setup device
    device_type, available_gpus = setup_device()
    
    if force_cpu or device_type == 'cpu':
        # Train on CPU
        losses = train_single_gpu(txt_path, epochs, batch_size, seq_len, lr, 'cpu',
                                max_model_length, stride)
    elif available_gpus == 1 or num_gpus == 1:
        # Train on single GPU
        losses = train_single_gpu(txt_path, epochs, batch_size, seq_len, lr, 'cuda',
                                max_model_length, stride)
    else:
        # Train on multiple GPUs
        world_size = num_gpus if num_gpus is not None else available_gpus
        world_size = min(world_size, available_gpus)
        
        print(f"Starting distributed training on {world_size} GPUs")
        
        # Use multiprocessing to spawn processes for each GPU
        mp.spawn(train_distributed_gpu, 
                args=(world_size, txt_path, epochs, batch_size, seq_len, lr, max_model_length, stride),
                nprocs=world_size, 
                join=True)
        
        losses = []  # Losses are handled in distributed function
    
    print(f"Training completed at {datetime.now()}")
    return losses

def main():
    parser = argparse.ArgumentParser(description='Train Question Generation Model')
    parser.add_argument('--txt_path', type=str, default="data/questions.txt",
                        help='Path to training text file')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--num_gpus', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--max_model_length', type=int, default=1024, 
                        help='Maximum sequence length the model can handle')
    parser.add_argument('--stride', type=int, default=None,
                        help='Step size for sliding window (default: no overlap)')
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    try:
        train_question_gen_model(
            txt_path=args.txt_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            force_cpu=args.force_cpu,
            num_gpus=args.num_gpus,
            max_model_length=args.max_model_length,
            stride=args.stride
        )
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()