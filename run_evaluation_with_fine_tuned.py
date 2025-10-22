#!/usr/bin/env python3
"""
Run baseline evaluation with the fine-tuned Qwen2-VL model
"""
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm
from itertools import combinations

class FineTunedSafetyEvaluator:
    def __init__(self, model_path: str, device: str = "cpu"):
        print(f"Loading fine-tuned model from {model_path}...")
        
        # Load base model
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Load fine-tuned LoRA weights
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded on {device}")
    
    def compare_two_images(self, img1_path: str, img2_path: str, prompt: str) -> str:
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            text = f"<|image_pad|><|image_pad|>{prompt}"
            
            inputs = self.processor(
                text=text,
                images=[img1, img2],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
    
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            input_length = len(inputs['input_ids'][0])
            response = self.processor.tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            return f"Error comparing images: {str(e)}"

def run_pairwise_evaluation(output_file: str = "results/fine_tuned/pairwise_results.json"):
    evaluator = FineTunedSafetyEvaluator(
        model_path="models/fine_tuned_qwen2vl",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Get all sample images
    sample_dir = Path("data/samples/Sample SVI")
    image_files = sorted(sample_dir.glob("*.jpg"))
    
    print(f"Found {len(image_files)} images")
    print("Running pairwise comparisons...")
    
    # Generate all pairs
    pairs = list(combinations(image_files, 2))
    print(f"Total comparisons: {len(pairs)}")
    
    prompt = "Which image shows a safer street environment? Compare the safety features."
    
    results = []
    for img1, img2 in tqdm(pairs, desc="Comparing pairs"):
        response = evaluator.compare_two_images(str(img1), str(img2), prompt)
        
        results.append({
            "image1": img1.name,
            "image2": img2.name,
            "prompt": prompt,
            "response": response
        })
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    print(f"Total comparisons: {len(results)}")

if __name__ == "__main__":
    run_pairwise_evaluation()

