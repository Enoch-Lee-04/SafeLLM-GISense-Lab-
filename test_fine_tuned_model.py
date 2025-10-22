#!/usr/bin/env python3
"""
Test the fine-tuned Qwen2-VL model on sample images
"""
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

def test_fine_tuned_model():
    print("🔄 Loading fine-tuned model...")
    
    # Load the base model
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load the fine-tuned LoRA weights
    model = PeftModel.from_pretrained(
        base_model,
        "models/fine_tuned_qwen2vl"
    )
    
    processor = AutoProcessor.from_pretrained(
        "models/fine_tuned_qwen2vl",
        trust_remote_code=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Model loaded on: {device}")
    
    # Test on a sample image
    sample_image_path = "data/samples/Sample SVI/1.jpg"
    
    if not Path(sample_image_path).exists():
        print(f"❌ Sample image not found: {sample_image_path}")
        return
    
    image = Image.open(sample_image_path).convert("RGB")
    
    # Test prompt
    prompt = "Assess the safety risks in this street view image."
    text_with_image = f"<|image_pad|>{prompt}"
    
    print(f"\n📸 Testing on: {sample_image_path}")
    print(f"❓ Prompt: {prompt}")
    
    # Process input
    inputs = processor(
        text=text_with_image,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    # Generate response
    print("\n🤖 Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
    
    # Decode response
    input_length = len(inputs['input_ids'][0])
    response = processor.tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )
    
    print("\n" + "="*60)
    print("📝 FINE-TUNED MODEL RESPONSE:")
    print("="*60)
    print(response)
    print("="*60)
    
    # Compare with base model (optional)
    print("\n\n🔄 Loading base model for comparison...")
    base_model_only = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor_base = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True
    )
    
    inputs_base = processor_base(
        text=text_with_image,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    print("🤖 Generating base model response...")
    with torch.no_grad():
        outputs_base = base_model_only.generate(
            **inputs_base,
            max_new_tokens=200,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor_base.tokenizer.eos_token_id,
            eos_token_id=processor_base.tokenizer.eos_token_id
        )
    
    input_length_base = len(inputs_base['input_ids'][0])
    response_base = processor_base.tokenizer.decode(
        outputs_base[0][input_length_base:],
        skip_special_tokens=True
    )
    
    print("\n" + "="*60)
    print("📝 BASE MODEL RESPONSE (for comparison):")
    print("="*60)
    print(response_base)
    print("="*60)
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    test_fine_tuned_model()

