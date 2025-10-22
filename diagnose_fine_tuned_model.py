#!/usr/bin/env python3
"""
Diagnose the fine-tuned model to check if it's working correctly
"""
import torch
from pathlib import Path
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
import json

def diagnose_model():
    model_path = "models/fine_tuned_qwen2vl"
    
    print("🔍 Diagnosing fine-tuned model...\n")
    
    # Check if files exist
    print("1. Checking model files...")
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_path}")
        return
    
    files = list(model_dir.glob("*"))
    print(f"✅ Found {len(files)} files in model directory:")
    for f in sorted(files)[:10]:
        print(f"   - {f.name}")
    
    # Check adapter config
    print("\n2. Checking adapter configuration...")
    adapter_config = model_dir / "adapter_config.json"
    if adapter_config.exists():
        with open(adapter_config) as f:
            config = json.load(f)
        print(f"✅ LoRA rank (r): {config.get('r')}")
        print(f"✅ LoRA alpha: {config.get('lora_alpha')}")
        print(f"✅ Target modules: {config.get('target_modules')}")
    else:
        print("❌ adapter_config.json not found!")
    
    # Try loading the model
    print("\n3. Loading model...")
    try:
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        print("✅ Base model loaded")
        
        model = PeftModel.from_pretrained(base_model, model_path)
        print("✅ LoRA adapter loaded")
        
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print("✅ Processor loaded")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Check tokenizer
    print("\n4. Checking tokenizer...")
    test_text = "Test tokenization"
    tokens = processor.tokenizer(test_text, return_tensors="pt")
    decoded = processor.tokenizer.decode(tokens['input_ids'][0])
    print(f"✅ Tokenization working: '{test_text}' -> {tokens['input_ids'].shape}")
    
    # Test simple generation (no image)
    print("\n5. Testing text-only generation...")
    try:
        simple_input = processor.tokenizer("Hello, how are", return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**simple_input, max_new_tokens=10)
        decoded_output = processor.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"✅ Generated: '{decoded_output}'")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
    
    # Check if model is actually in eval mode
    print("\n6. Checking model state...")
    print(f"✅ Model is in {'eval' if not model.training else 'training'} mode")
    
    # Check trainable parameters
    print("\n7. Checking LoRA parameters...")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Trainable params: {trainable_params:,}")
    print(f"✅ Total params: {total_params:,}")
    print(f"✅ Trainable %: {100 * trainable_params / total_params:.4f}%")
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    diagnose_model()

