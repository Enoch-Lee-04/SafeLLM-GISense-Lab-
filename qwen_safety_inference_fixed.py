"""
Fixed Qwen VLM Safety Assessment Inference
Use fine-tuned Qwen model for street view safety assessment with correct processing pipeline
"""

import os
import json
import torch
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoProcessor, 
    Qwen2VLForConditionalGeneration
)


class FixedSafetyAssessmentModel:
    """Fixed safety assessment model using fine-tuned Qwen-VL"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the safety assessment model
        
        Args:
            model_path: Path to fine-tuned model
            device: Device to run inference on
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        print(f"Loading model from {model_path} on {self.device}")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def assess_safety(
        self, 
        image_path: str, 
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        Assess safety of a street view image using correct processing pipeline
        
        Args:
            image_path: Path to street view image
            prompt: Safety assessment prompt
            max_length: Maximum response length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Safety assessment response
        """
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Create text with image token placeholder
            # The processor will automatically replace <|image_pad|> with the correct number of tokens
            text_with_image = f"<|image_pad|>{prompt}"
            
            # Process inputs using the correct method
            inputs = self.processor(
                text=text_with_image,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.processor.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Extract only the assistant's response (remove input text)
            input_length = len(inputs['input_ids'][0])
            response = self.processor.tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            return f"Error processing image: {str(e)}"
    
    def compare_two_images(self, img1_path: str, img2_path: str, prompt: str) -> str:
        """
        Compare two images side by side using correct processing pipeline
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image  
            prompt: Comparison prompt
            
        Returns:
            Comparison response
        """
        try:
            # Load images
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            # Create text with image token placeholders
            # The processor will automatically handle multiple images
            text_with_images = f"<|image_pad|><|image_pad|>{prompt}"
            
            # Process inputs using the correct method
            inputs = self.processor(
                text=text_with_images,
                images=[img1, img2],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            input_length = len(inputs['input_ids'][0])
            response = self.processor.tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            return f"Error comparing images: {str(e)}"
    
    def batch_assess_safety(
        self, 
        image_paths: List[str], 
        prompt: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Assess safety for multiple images
        
        Args:
            image_paths: List of image paths
            prompt: Safety assessment prompt
            **kwargs: Additional arguments for assess_safety
            
        Returns:
            List of assessment results
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            response = self.assess_safety(image_path, prompt, **kwargs)
            
            results.append({
                'image_path': image_path,
                'prompt': prompt,
                'response': response,
                'image_id': os.path.basename(image_path)
            })
        
        return results


def create_safety_prompts() -> Dict[str, str]:
    """Create different types of safety assessment prompts"""
    return {
        'safety_score': """Analyze this street view image and provide a safety score from 1-10 (10 being safest).

Consider:
- Sidewalk condition and width
- Traffic density and speed
- Street lighting
- Road maintenance
- Pedestrian-vehicle separation
- Visibility and sightlines

Format your response as:
SAFETY SCORE: [1-10]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Brief explanation]""",

        'binary_classification': """Classify this street view image as SAFE or UNSAFE for pedestrians.

SAFE = Good sidewalks, low traffic, good lighting, well-maintained infrastructure
UNSAFE = Poor sidewalks, heavy traffic, poor lighting, infrastructure issues

Response format:
CLASSIFICATION: [SAFE/UNSAFE]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASON: [Brief explanation]""",

        'detailed_analysis': """Provide a detailed safety analysis of this street view image.

Evaluate these aspects (1-10 scale each):
1. Pedestrian Safety: ___
2. Traffic Safety: ___
3. Lighting Safety: ___
4. Infrastructure Safety: ___
5. Crime Safety: ___

Overall Score: ___/50
Main Concerns: [List top 3 issues]
Strengths: [List top 3 positive features]""",

        'risk_assessment': """Assess the safety risks in this street view image.

Identify risks by category:
HIGH RISK: [List high-risk elements]
MEDIUM RISK: [List medium-risk elements]
LOW RISK: [List low-risk elements]

Overall Risk Level: [LOW/MEDIUM/HIGH]
Primary Risk Factor: [Main safety concern]""",

        'comparison': """Compare these two street view images and determine which one is SAFER for pedestrians.

Consider these factors:
- Sidewalk condition and width
- Traffic density and speed  
- Street lighting quality
- Road maintenance
- Pedestrian-vehicle separation
- Visibility and sight lines
- Overall infrastructure safety

Provide your response as:
SAFER IMAGE: [1 or 2]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Brief explanation of why one is safer than the other]"""
    }


def main():
    parser = argparse.ArgumentParser(description="Fixed Qwen VLM Safety Assessment Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--image_path", type=str, help="Path to single image")
    parser.add_argument("--image_folder", type=str, help="Folder containing images")
    parser.add_argument("--prompt_type", type=str, default="safety_score", 
                       choices=['safety_score', 'binary_classification', 'detailed_analysis', 'risk_assessment', 'comparison'],
                       help="Type of safety assessment prompt")
    parser.add_argument("--custom_prompt", type=str, help="Custom prompt text")
    parser.add_argument("--output_file", type=str, help="Output file for results")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum response length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Initialize model
    model = FixedSafetyAssessmentModel(args.model_path, args.device)
    
    # Get prompt
    if args.custom_prompt:
        prompt = args.custom_prompt
    else:
        prompts = create_safety_prompts()
        prompt = prompts[args.prompt_type]
    
    print(f"Using prompt type: {args.prompt_type}")
    print(f"Prompt: {prompt[:100]}...")
    
    # Get image paths
    image_paths = []
    
    if args.image_path:
        if os.path.exists(args.image_path):
            image_paths.append(args.image_path)
        else:
            print(f"Error: Image {args.image_path} not found!")
            return
    
    elif args.image_folder:
        if os.path.exists(args.image_folder):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_paths.extend(Path(args.image_folder).glob(ext))
                image_paths.extend(Path(args.image_folder).glob(ext.upper()))
            image_paths = [str(p) for p in image_paths]
        else:
            print(f"Error: Folder {args.image_folder} not found!")
            return
    
    else:
        print("Error: Must provide either --image_path or --image_folder!")
        return
    
    if not image_paths:
        print("No images found!")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Run inference
    if len(image_paths) == 1:
        # Single image
        response = model.assess_safety(
            image_paths[0], 
            prompt,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        print(f"\nImage: {os.path.basename(image_paths[0])}")
        print(f"Assessment: {response}")
        
        results = [{
            'image_path': image_paths[0],
            'prompt': prompt,
            'response': response
        }]
    
    else:
        # Multiple images
        results = model.batch_assess_safety(
            image_paths,
            prompt,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        # Print results
        print("\n" + "="*60)
        print("SAFETY ASSESSMENT RESULTS")
        print("="*60)
        
        for result in results:
            print(f"\nImage: {result['image_id']}")
            print(f"Assessment: {result['response']}")
            print("-" * 40)
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
