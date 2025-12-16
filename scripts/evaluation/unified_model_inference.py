#!/usr/bin/env python3
"""
Unified inference interface for GPT-4o-mini (fine-tuned) and Qwen2-VL (fine-tuned)
Provides a common API for both models
"""

import torch
import json
from pathlib import Path
from PIL import Image
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import base64
import io

# OpenAI imports
from openai import OpenAI

# Transformers imports for Qwen
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel


class BaseModelInference(ABC):
    """Base class for model inference"""
    
    @abstractmethod
    def predict(self, image_path: str, prompt: str) -> str:
        """
        Generate prediction for an image
        
        Args:
            image_path: Path to the image
            prompt: Text prompt for the model
            
        Returns:
            Model's response as string
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name"""
        pass


class GPT4oMiniFineTunedInference(BaseModelInference):
    """Inference for fine-tuned GPT-4o-mini"""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize fine-tuned GPT-4o-mini inference
        
        Args:
            model_name: Fine-tuned model name (e.g., "ft:gpt-4o-mini-2024-07-18:...")
            api_key: OpenAI API key
        """
        if api_key is None:
            api_key_path = Path(__file__).parent.parent.parent / "API_KEY_Enoch"
            with open(api_key_path, "r") as f:
                api_key = f.read().strip()
        
        self.client = OpenAI(api_key=api_key)
        
        # Load model name from saved info if not provided
        if model_name is None:
            info_path = Path(__file__).parent.parent.parent / "models" / "gpt4o_mini_finetuned" / "fine_tune_info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    model_name = info.get('model_name')
        
        if model_name is None:
            raise ValueError("Fine-tuned model name not provided and not found in saved info")
        
        self.model = model_name
        print(f"[OK] Initialized fine-tuned GPT-4o-mini: {self.model}")
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def predict(self, image_path: str, prompt: str) -> str:
        """Generate prediction using fine-tuned GPT-4o-mini"""
        # Updated system message for consistent evaluation
        system_message = """You are a Vision Language Model trained to evaluate the perceived safety of a single street-view image.

Your task is to analyze one image at a time and predict a safety score from 0 to 10:

1. A **safety score from 0 to 10**, where:
   - 0 means extremely unsafe
   - 10 means extremely safe

2. After providing the score, explain your reasoning in **2–5 concise bullet points**, each grounded only in **visible environmental cues** present in the image. Base your assessment entirely on what is visible — do not infer details beyond the image.

Visual elements to consider:
- Lighting and visibility: streetlights, shadows, time of day, visibility distance
- Maintenance and cleanliness: litter, graffiti, broken infrastructure, general upkeep
- Social activity: presence of people, pedestrians, vehicles, or community engagement
- Environment and design: open vs. enclosed areas, fencing, barriers, accessibility
- Safety features: crosswalks, sidewalks, cameras, signage, traffic control

Respond using this exact format:
Score: X/10
Reason: ["reason 1", "reason 2", "reason 3", ...]

Be specific and consistent. Avoid vague or overly generic explanations. Use the full 0–10 scale when appropriate."""
        
        # For text-only fine-tuned models, we might need to adjust
        # Check if the model supports vision or if we need text-only format
        
        try:
            # Try with vision (if fine-tuned with images)
            base64_image = self.encode_image(image_path)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
        except Exception as e:
            # Fallback to text-only
            print(f"[WARNING] Vision inference failed, using text-only format: {e}")
            image_filename = Path(image_path).name
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nImage: {image_filename}"
                    }
                ],
                max_tokens=500
            )
        
        return response.choices[0].message.content
    
    def get_model_name(self) -> str:
        return f"GPT-4o-mini (fine-tuned)"


class Qwen2VLFineTunedInference(BaseModelInference):
    """Inference for fine-tuned Qwen2-VL"""
    
    def __init__(
        self, 
        model_path: str = None,
        base_model: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "auto"
    ):
        """
        Initialize fine-tuned Qwen2-VL inference
        
        Args:
            model_path: Path to fine-tuned LoRA weights
            base_model: Base model identifier
            device: Device to use (auto, cuda, cpu)
        """
        if model_path is None:
            model_path = str(Path(__file__).parent.parent.parent / "models" / "fine_tuned_qwen2vl")
        
        self.model_path = model_path
        self.base_model = base_model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading Qwen2-VL from {model_path}...")
        
        # Load base model
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Load fine-tuned LoRA weights
        self.model = PeftModel.from_pretrained(base, model_path)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        self.model.eval()
        print(f"[OK] Initialized Qwen2-VL (fine-tuned) on {self.device}")
    
    def predict(self, image_path: str, prompt: str) -> str:
        """Generate prediction using fine-tuned Qwen2-VL"""
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Updated system prompt for consistent evaluation
        system_prompt = """You are a Vision Language Model trained to evaluate the perceived safety of a single street-view image.

Your task is to analyze one image at a time and predict a safety score from 0 to 10:

1. A **safety score from 0 to 10**, where:
   - 0 means extremely unsafe
   - 10 means extremely safe

2. After providing the score, explain your reasoning in **2–5 concise bullet points**, each grounded only in **visible environmental cues** present in the image. Base your assessment entirely on what is visible — do not infer details beyond the image.

Visual elements to consider:
- Lighting and visibility: streetlights, shadows, time of day, visibility distance
- Maintenance and cleanliness: litter, graffiti, broken infrastructure, general upkeep
- Social activity: presence of people, pedestrians, vehicles, or community engagement
- Environment and design: open vs. enclosed areas, fencing, barriers, accessibility
- Safety features: crosswalks, sidewalks, cameras, signage, traffic control

Respond using this exact format:
Score: X/10
Reason: ["reason 1", "reason 2", "reason 3", ...]

Be specific and consistent. Avoid vague or overly generic explanations. Use the full 0–10 scale when appropriate."""
        
        # Use chat template format (matching training format)
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template with generation prompt
        text = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # Important for inference!
        )
        
        # Process input with chat template
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response with strict controls
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,  # Increased to accommodate the structured format
                do_sample=False,
                repetition_penalty=1.5,  # Stronger penalty
                num_beams=1,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode response (skip input tokens)
        input_length = len(inputs['input_ids'][0])
        response = self.processor.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
        
        # Clean up special tokens and artifacts
        response = response.replace('<|im_end|>', '').replace('<|endoftext|>', '')
        response = response.replace('<div>', '').replace('</div>', '')
        
        return response.strip()
    
    def compare_two_images(self, img1_path: str, img2_path: str, prompt: str) -> str:
        """
        Compare two images side by side using chat template format
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            prompt: Comparison prompt
            
        Returns:
            Comparison response
        """
        try:
            # Load images
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
            
            # Use chat template format with multiple images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template with generation prompt
            text = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process input with chat template (multiple images)
            inputs = self.processor(
                text=text,
                images=[img1, img2],
                return_tensors="pt",
                padding=True
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
                    do_sample=False,
                    repetition_penalty=1.5,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response (skip input tokens)
            input_length = len(inputs['input_ids'][0])
            response = self.processor.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
            
            # Clean up special tokens and artifacts
            response = response.replace('<|im_end|>', '').replace('<|endoftext|>', '')
            response = response.replace('<div>', '').replace('</div>', '')
            
            return response.strip()
            
        except Exception as e:
            return f"Error comparing images: {str(e)}"
    
    def get_model_name(self) -> str:
        return "Qwen2-VL-2B (fine-tuned)"


class ModelFactory:
    """Factory to create model inference instances"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModelInference:
        """
        Create a model inference instance
        
        Args:
            model_type: Type of model ("gpt4o-mini-ft", "qwen2vl-ft")
            **kwargs: Additional arguments for the model
            
        Returns:
            Model inference instance
        """
        model_type = model_type.lower()
        
        if model_type == "gpt4o-mini-ft":
            return GPT4oMiniFineTunedInference(**kwargs)
        elif model_type == "qwen2vl-ft":
            return Qwen2VLFineTunedInference(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_available_models() -> list:
        """Return list of available model types"""
        return ["gpt4o-mini-ft", "qwen2vl-ft"]


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


def test_inference():
    """Test function to verify all models work"""
    print("=" * 60)
    print("TESTING MODEL INFERENCE")
    print("=" * 60)
    
    # Test image
    test_image = Path(__file__).parent.parent.parent / "data" / "images" / "image" / "1.jpg"
    if not test_image.exists():
        print(f"[ERROR] Test image not found: {test_image}")
        return
    
    test_prompt = "Analyze this street view image and provide a safety score from 1-10 (10 being safest)."
    
    # Test each model
    models_to_test = [
        ("gpt4o-mini-ft", {}),
        ("qwen2vl-ft", {}),
    ]
    
    for model_type, kwargs in models_to_test:
        try:
            print(f"\n{'=' * 60}")
            print(f"Testing: {model_type}")
            print(f"{'=' * 60}")
            
            model = ModelFactory.create_model(model_type, **kwargs)
            response = model.predict(str(test_image), test_prompt)
            
            print(f"\nResponse from {model.get_model_name()}:")
            print("-" * 60)
            print(response)
            print("-" * 60)
            
        except Exception as e:
            print(f"[ERROR] Error testing {model_type}: {e}")
    
    print("\n[OK] Testing complete!")


if __name__ == "__main__":
    test_inference()


