import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from models.transformer import TransformerModel
import argparse

def test_model_dynamic_shapes(model, vocab_size=50257):
    """Test the model with different sequence lengths"""
    model.eval()
    test_lengths = [1, 4, 8, 16, 32, 64, 128]
    
    print("Testing model with different sequence lengths...")
    for seq_len in test_lengths:
        try:
            dummy_input = torch.randint(0, vocab_size, (1, seq_len))
            with torch.no_grad():
                output = model(dummy_input)
            print(f"✓ Sequence length {seq_len}: input {dummy_input.shape} -> output {output.shape}")
        except Exception as e:
            print(f"✗ Sequence length {seq_len} failed: {e}")
            return False
    return True

def export_model_to_onnx(model_path="models/transformer_lm.pth", 
                        output_path="transformer_mhla.onnx",
                        vocab_size=50257,
                        test_dynamic=True):
    """
    Export PyTorch model to ONNX with dynamic sequence length support
    """
    
    # Initialize model with same parameters as training
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=512,
        latent_layers=2,
        latent_length=64
    )
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded model weights from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    model.eval()
    
    # Test dynamic shapes if requested
    if test_dynamic:
        if not test_model_dynamic_shapes(model, vocab_size):
            print("Model failed dynamic shape testing")
            return False
    
    # Use a medium-sized input for export (this will be the example shape)
    export_seq_len = 32
    dummy_input = torch.randint(0, vocab_size, (1, export_seq_len))
    print(f"Dummy input shape: {dummy_input.shape}")
    
    # Test the model with dummy input first
    try:
        with torch.no_grad():
            test_output = model(dummy_input)
        print(f"Model test output shape: {test_output.shape}")
    except Exception as e:
        print(f"Model test failed: {e}")
        return False
    
    # Export to ONNX with dynamic axes
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "seq_len"},
                "output": {0: "batch_size", 1: "seq_len"},
            },
            opset_version=16,
            do_constant_folding=True,
            verbose=False,
            export_params=True,
        )
        print(f"Successfully exported model to {output_path}")
        
        # Verify the exported model with different sequence lengths
        import onnxruntime as ort
        
        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        
        # Get input/output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"ONNX Model Info:")
        print(f"  Input: {input_info.name}, shape: {input_info.shape}")
        print(f"  Output: {output_info.name}, shape: {output_info.shape}")
        
        # Test with different sequence lengths
        test_lengths = [1, 4, 8, 16, 32, 64]
        print("\nTesting ONNX model with different sequence lengths...")
        
        for seq_len in test_lengths:
            try:
                test_input = torch.randint(0, vocab_size, (1, seq_len)).numpy().astype('int64')
                onnx_output = session.run(["output"], {"input": test_input})
                print(f"✓ ONNX seq_len {seq_len}: input {test_input.shape} -> output {onnx_output[0].shape}")
            except Exception as e:
                print(f"✗ ONNX seq_len {seq_len} failed: {e}")
                return False
        
        print("ONNX dynamic export verification successful!")
        return True
        
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return False

def quantize_model(input_path="transformer_mhla.onnx", 
                  output_path="transformer_mhla_quant.onnx"):
    """
    Quantize the ONNX model for better performance
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QUInt8
        )
        print(f"Model quantized and saved to {output_path}")
        
        # Verify quantized model with dynamic shapes
        import onnxruntime as ort
        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        
        # Test with different sequence lengths
        test_lengths = [1, 4, 8, 16, 32]
        print("Testing quantized model with different sequence lengths...")
        
        for seq_len in test_lengths:
            try:
                dummy_input = torch.randint(0, 50257, (1, seq_len)).numpy().astype('int64')
                output = session.run(["output"], {"input": dummy_input})
                print(f"✓ Quantized model seq_len {seq_len}: {dummy_input.shape} -> {output[0].shape}")
            except Exception as e:
                print(f"✗ Quantized model seq_len {seq_len} failed: {e}")
                return False
        
        return True
        
    except ImportError:
        print("onnxruntime quantization not available. Install with: pip install onnxruntime[quantization]")
        return False
    except Exception as e:
        print(f"Quantization failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Export and quantize transformer model to ONNX')
    parser.add_argument('--model_path', type=str, default="models/transformer_lm.pth",
                        help='Path to trained PyTorch model')
    parser.add_argument('--output_path', type=str, default="transformer_mhla.onnx",
                        help='Output path for ONNX model')
    parser.add_argument('--quantized_path', type=str, default="transformer_mhla_quant.onnx",
                        help='Output path for quantized ONNX model')
    parser.add_argument('--vocab_size', type=int, default=50257,
                        help='Vocabulary size')
    parser.add_argument('--no_quantize', action='store_true',
                        help='Skip quantization step')
    parser.add_argument('--no_test', action='store_true',
                        help='Skip dynamic shape testing')
    
    args = parser.parse_args()
    
    print("Starting ONNX export process with dynamic sequence length support...")
    
    # Export to ONNX
    if export_model_to_onnx(args.model_path, args.output_path, args.vocab_size, not args.no_test):
        print("✓ ONNX export successful")
        
        # Quantize if requested
        if not args.no_quantize:
            print("\nStarting quantization...")
            if quantize_model(args.output_path, args.quantized_path):
                print("✓ Quantization successful")
                print(f"Both models support dynamic sequence lengths:")
                print(f"  - Standard: {args.output_path}")
                print(f"  - Quantized: {args.quantized_path}")
            else:
                print("✗ Quantization failed")
        else:
            print("Skipping quantization")
            print(f"Model with dynamic sequence length support: {args.output_path}")
            
    else:
        print("✗ ONNX export failed")

if __name__ == "__main__":
    main()