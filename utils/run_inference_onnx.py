import argparse
import onnxruntime as ort
import numpy as np
import torch
from tokenizers import Tokenizer

def load_tokenizer(tokenizer_path="tokenizer/tokenizer.json"):
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

def run_inference(input_text, tokenizer_path="tokenizer/tokenizer.json", onnx_model="transformer_mhla.onnx", 
                 max_len=None, generation_length=10, temperature=1.0, top_k=50):
    """
    Run inference with dynamic sequence length support
    
    Args:
        input_text: Input prompt text
        tokenizer_path: Path to tokenizer file
        onnx_model: Path to ONNX model
        max_len: Maximum sequence length (None for no limit up to model max)
        generation_length: Number of tokens to generate
        temperature: Sampling temperature (1.0 = no change, <1.0 = more focused, >1.0 = more random)
        top_k: Top-k sampling (0 = disabled)
    """
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer is None:
        return None
    
    # Tokenize input
    encoded = tokenizer.encode(input_text)
    input_ids = encoded.ids
    
    print(f"Input text: '{input_text}'")
    print(f"Tokenized to {len(input_ids)} tokens: {input_ids}")
    
    # Handle sequence length constraints
    original_length = len(input_ids)
    
    if len(input_ids) == 0:
        print("Empty input, using a default token")
        input_ids = [0]  # Use a default token
    
    if max_len is not None and len(input_ids) > max_len:
        print(f"Input too long ({len(input_ids)} tokens), truncating to {max_len}")
        input_ids = input_ids[:max_len]
        original_length = len(input_ids)
    
    print(f"Processing input length: {len(input_ids)} tokens")
    
    # Load ONNX session
    try:
        session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
        
        # Get model info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"Model input info: {input_info.name}, shape: {input_info.shape}")
        print(f"Model output info: {output_info.name}, shape: {output_info.shape}")
        
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return None
    
    # Generate tokens
    generated_tokens = input_ids.copy()
    
    for step in range(generation_length):
        # Create input array for current sequence
        current_input = np.array([generated_tokens], dtype=np.int64)
        
        try:
            # Run inference
            outputs = session.run(output_names=["output"], input_feed={"input": current_input})
            logits = outputs[0]  # (batch_size, seq_len, vocab_size)
            
            # Get logits for the last token
            last_token_logits = logits[0, -1, :]  # (vocab_size,)
            
            # Apply temperature
            if temperature != 1.0:
                last_token_logits = last_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                # Get the top-k logits
                top_k_logits, top_k_indices = torch.topk(torch.tensor(last_token_logits), top_k)
                # Create a mask for non-top-k tokens
                mask = torch.full_like(torch.tensor(last_token_logits), float('-inf'))
                mask[top_k_indices] = top_k_logits
                last_token_logits = mask.numpy()
            
            # Convert to probabilities
            # Handle potential overflow/underflow
            last_token_logits = np.clip(last_token_logits, -100, 100)
            exp_logits = np.exp(last_token_logits - np.max(last_token_logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            # Sample next token
            if temperature == 0.0:
                # Greedy sampling
                next_token_id = np.argmax(probabilities)
            else:
                # Multinomial sampling
                try:
                    next_token_id = np.random.choice(len(probabilities), p=probabilities)
                except:
                    # Fallback to greedy if sampling fails
                    next_token_id = np.argmax(probabilities)
            
            generated_tokens.append(int(next_token_id))
            
            print(f"Step {step + 1}: Generated token {next_token_id} (prob: {probabilities[next_token_id]:.4f})")
            
            # Optional: Stop if we hit an end token or special stopping condition
            # You can add your stopping criteria here
            
        except Exception as e:
            print(f"Error in generation step {step + 1}: {e}")
            break
    
    # Decode the results
    try:
        # Decode only the original input
        if original_length > 0:
            original_decoded = tokenizer.decode(input_ids[:original_length])
        else:
            original_decoded = ""
        
        # Decode the full generated sequence
        full_decoded = tokenizer.decode(generated_tokens)
        
        # Extract just the generated part
        if len(generated_tokens) > original_length:
            generated_part = tokenizer.decode(generated_tokens[original_length:])
        else:
            generated_part = ""
        
        print(f"\nResults:")
        print(f"Original input: '{original_decoded}'")
        print(f"Generated continuation: '{generated_part}'")
        print(f"Full sequence: '{full_decoded}'")
        
        return full_decoded
        
    except Exception as e:
        print(f"Error decoding tokens: {e}")
        # Fallback: return raw token IDs
        return f"Generated tokens: {generated_tokens}"

def validate_model(onnx_model, vocab_size=50257):
    """Validate that the model works with different sequence lengths"""
    try:
        session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
        
        # Get model info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"Model validation:")
        print(f"  Input: {input_info.name}, shape: {input_info.shape}")
        print(f"  Output: {output_info.name}, shape: {output_info.shape}")
        
        # Test different sequence lengths
        test_lengths = [1, 4, 8, 16, 32, 64, 128]
        
        for seq_len in test_lengths:
            try:
                # Create random input
                test_input = np.random.randint(0, min(vocab_size, 1000), (1, seq_len), dtype=np.int64)
                
                # Run inference
                outputs = session.run(output_names=["output"], input_feed={"input": test_input})
                output_shape = outputs[0].shape
                
                print(f"  ✓ Length {seq_len:3d}: {test_input.shape} -> {output_shape}")
                
                # Verify output shape is correct
                expected_shape = (1, seq_len, vocab_size)
                if output_shape != expected_shape:
                    print(f"    Warning: Expected {expected_shape}, got {output_shape}")
                    
            except Exception as e:
                print(f"  ✗ Length {seq_len:3d}: Failed - {e}")
                return False
                
        print("Model validation successful!")
        return True
        
    except Exception as e:
        print(f"Model validation failed: {e}")
        return False

def benchmark_model(onnx_model, vocab_size=50257, num_runs=10):
    """Benchmark the model with different sequence lengths"""
    import time
    
    try:
        session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
        test_lengths = [1, 8, 16, 32, 64, 128]
        
        print(f"\nBenchmarking model performance ({num_runs} runs per length):")
        print(f"{'Length':>6} {'Avg Time (ms)':>15} {'Tokens/sec':>12}")
        print("-" * 35)
        
        for seq_len in test_lengths:
            times = []
            
            for _ in range(num_runs):
                test_input = np.random.randint(0, min(vocab_size, 1000), (1, seq_len), dtype=np.int64)
                
                start_time = time.time()
                outputs = session.run(output_names=["output"], input_feed={"input": test_input})
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            avg_time = np.mean(times) * 1000  # Convert to milliseconds
            tokens_per_sec = seq_len / (np.mean(times)) if np.mean(times) > 0 else 0
            
            print(f"{seq_len:6d} {avg_time:13.2f} {tokens_per_sec:10.1f}")
            
    except Exception as e:
        print(f"Benchmarking failed: {e}")

def interactive_mode(tokenizer_path="tokenizer/tokenizer.json", onnx_model="transformer_mhla.onnx"):
    """Interactive text generation mode"""
    print("Interactive text generation mode. Type 'quit' to exit.")
    print("Available commands:")
    print("  !temp <value>  - Set temperature (default: 1.0)")
    print("  !topk <value>  - Set top-k sampling (default: 50)")
    print("  !len <value>   - Set generation length (default: 10)")
    print("  !help          - Show this help")
    print()
    
    # Default parameters
    temperature = 1.0
    top_k = 50
    generation_length = 10
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.startswith('!temp '):
                try:
                    temperature = float(user_input[6:])
                    print(f"Temperature set to {temperature}")
                except ValueError:
                    print("Invalid temperature value")
                continue
            elif user_input.startswith('!topk '):
                try:
                    top_k = int(user_input[6:])
                    print(f"Top-k set to {top_k}")
                except ValueError:
                    print("Invalid top-k value")
                continue
            elif user_input.startswith('!len '):
                try:
                    generation_length = int(user_input[5:])
                    print(f"Generation length set to {generation_length}")
                except ValueError:
                    print("Invalid generation length")
                continue
            elif user_input == '!help':
                print("Available commands:")
                print("  !temp <value>  - Set temperature (current:", temperature, ")")
                print("  !topk <value>  - Set top-k sampling (current:", top_k, ")")
                print("  !len <value>   - Set generation length (current:", generation_length, ")")
                print("  !help          - Show this help")
                continue
            elif user_input.startswith('!'):
                print("Unknown command. Type !help for available commands.")
                continue
            
            if user_input:
                print(f"Generating with temp={temperature}, top_k={top_k}, length={generation_length}")
                print("-" * 50)
                result = run_inference(
                    user_input, 
                    tokenizer_path=tokenizer_path,
                    onnx_model=onnx_model,
                    generation_length=generation_length,
                    temperature=temperature,
                    top_k=top_k
                )
                print("-" * 50)
                print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run inference with ONNX transformer model')
    parser.add_argument('--model', type=str, default="transformer_mhla.onnx",
                        help='Path to ONNX model file')
    parser.add_argument('--tokenizer', type=str, default="tokenizer.json",
                        help='Path to tokenizer file')
    parser.add_argument('--text', type=str, 
                        help='Input text for generation')
    parser.add_argument('--length', type=int, default=10,
                        help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (0.0 = greedy, 1.0 = normal, >1.0 = more random)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling (0 = disabled)')
    parser.add_argument('--max_len', type=int,
                        help='Maximum input sequence length')
    parser.add_argument('--vocab_size', type=int, default=50257,
                        help='Vocabulary size for validation')
    parser.add_argument('--validate', action='store_true',
                        help='Validate model with different sequence lengths')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark model performance')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Validate model if requested
    if args.validate:
        print("Validating model...")
        if not validate_model(args.model, args.vocab_size):
            print("Model validation failed!")
            return
        print()
    
    # Benchmark model if requested
    if args.benchmark:
        benchmark_model(args.model, args.vocab_size)
        print()
    
    # Run interactive mode
    if args.interactive:
        interactive_mode(args.tokenizer, args.model)
        return
    
    # Single inference
    if args.text:
        print("Running single inference...")
        result = run_inference(
            args.text,
            tokenizer_path=args.tokenizer,
            onnx_model=args.model,
            max_len=args.max_len,
            generation_length=args.length,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        if result is None:
            print("Inference failed!")
        else:
            print(f"Final result: {result}")
    else:
        print("No text provided. Use --text 'your text here' or --interactive mode")
        print("Use --help for all options")

if __name__ == "__main__":
    main()