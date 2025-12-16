#!/usr/bin/env python3
"""
Prepare training data for OpenAI GPT-4o-mini fine-tuning
Converts ground truth data to OpenAI's JSONL format
"""

import json
from pathlib import Path
from typing import List, Dict
import base64
from PIL import Image
import io

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image to base64 string
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string
    """
    with Image.open(image_path) as img:
        # Resize if too large (OpenAI has size limits)
        max_size = (2000, 2000)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save to bytes
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
    return f"data:image/jpeg;base64,{img_str}"


def prepare_training_data(input_file: Path, output_file: Path, include_images: bool = False):
    """
    Convert ground truth data to OpenAI fine-tuning format with system message
    
    Args:
        input_file: Path to vlm_safety_training_data.json
        output_file: Path to save JSONL file
        include_images: Whether to include base64 encoded images (for vision models)
    """
    # System message for all training examples
    SYSTEM_MESSAGE = """You are a Vision Language Model trained to evaluate the perceived safety of a single street-view image.

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

    print(f"Loading ground truth data from {input_file}...")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} training examples")
    
    training_examples = []
    processed = 0
    errors = 0
    
    for item in data:
        try:
            image_path = item.get('image_path')
            prompt = item.get('prompt')
            expected_response = item.get('expected_response')
            task_type = item.get('task_type', 'unknown')
            
            if not all([image_path, prompt, expected_response]):
                print(f"[WARNING] Skipping item with missing fields")
                errors += 1
                continue
            
            # Create the training example in OpenAI format with system message
            messages = []
            
            if include_images and Path(image_path).exists():
                # For vision models (GPT-4o-mini supports vision)
                image_base64 = encode_image_to_base64(image_path)
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_MESSAGE
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_base64
                                }
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": expected_response
                    }
                ]
            else:
                # Text-only format (fallback)
                image_filename = Path(image_path).name if image_path else "unknown.jpg"
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_MESSAGE
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nImage: {image_filename}"
                    },
                    {
                        "role": "assistant",
                        "content": expected_response
                    }
                ]
            
            training_example = {
                "messages": messages
            }
            
            training_examples.append(training_example)
            processed += 1
            
            if processed % 10 == 0:
                print(f"Processed {processed}/{len(data)} examples...")
                
        except Exception as e:
            print(f"[ERROR] Error processing item: {e}")
            errors += 1
            continue
    
    # Save to JSONL format
    print(f"\nSaving {len(training_examples)} examples to {output_file}...")
    
    with open(output_file, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"[OK] Successfully created fine-tuning dataset!")
    print(f"   - Total examples: {len(training_examples)}")
    print(f"   - Errors: {errors}")
    print(f"   - Output: {output_file}")
    
    return training_examples


def create_validation_split(training_data: List[Dict], validation_ratio: float = 0.1):
    """
    Split data into training and validation sets
    
    Args:
        training_data: List of training examples
        validation_ratio: Ratio of data to use for validation
        
    Returns:
        Tuple of (train_data, val_data)
    """
    import random
    random.seed(42)
    
    # Shuffle data
    shuffled = training_data.copy()
    random.shuffle(shuffled)
    
    # Split
    split_idx = int(len(shuffled) * (1 - validation_ratio))
    train_data = shuffled[:split_idx]
    val_data = shuffled[split_idx:]
    
    return train_data, val_data


def main():
    """Main function to prepare OpenAI fine-tuning data"""
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / "configs" / "vlm_safety_training_data.json"
    
    # Output directory
    output_dir = project_root / "data" / "openai_finetuning"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if images should be included (vision fine-tuning)
    # Note: As of now, OpenAI's fine-tuning for vision might have limitations
    # We'll create both versions
    
    print("=" * 60)
    print("PREPARING OPENAI GPT-4o-mini FINE-TUNING DATA")
    print("=" * 60)
    
    # Version 1: With images (base64 encoded) - for vision fine-tuning
    print("\n[IMAGE] Creating version WITH images (base64 encoded)...")
    output_with_images = output_dir / "full_training_data_with_images.jsonl"
    
    try:
        training_data = prepare_training_data(
            input_file, 
            output_with_images, 
            include_images=True
        )
        
        # Create validation split
        train_data, val_data = create_validation_split(training_data)
        
        # Save splits
        train_output = output_dir / "train_with_images.jsonl"
        val_output = output_dir / "val_with_images.jsonl"
        
        with open(train_output, 'w') as f:
            for example in train_data:
                f.write(json.dumps(example) + '\n')
        
        with open(val_output, 'w') as f:
            for example in val_data:
                f.write(json.dumps(example) + '\n')
        
        print(f"   - Training set: {len(train_data)} examples -> {train_output}")
        print(f"   - Validation set: {len(val_data)} examples -> {val_output}")
        
    except Exception as e:
        print(f"[ERROR] Error creating version with images: {e}")
    
    # Version 2: Without images (text-only) - more reliable fallback
    print("\n[TEXT] Creating version WITHOUT images (text-only)...")
    output_without_images = output_dir / "training_data_text_only.jsonl"
    
    try:
        training_data_text = prepare_training_data(
            input_file, 
            output_without_images, 
            include_images=False
        )
        
        # Create validation split
        train_data_text, val_data_text = create_validation_split(training_data_text)
        
        # Save splits
        train_output_text = output_dir / "train_text_only.jsonl"
        val_output_text = output_dir / "val_text_only.jsonl"
        
        with open(train_output_text, 'w') as f:
            for example in train_data_text:
                f.write(json.dumps(example) + '\n')
        
        with open(val_output_text, 'w') as f:
            for example in val_data_text:
                f.write(json.dumps(example) + '\n')
        
        print(f"   - Training set: {len(train_data_text)} examples -> {train_output_text}")
        print(f"   - Validation set: {len(val_data_text)} examples -> {val_output_text}")
        
    except Exception as e:
        print(f"[ERROR] Error creating text-only version: {e}")
    
    print("\n" + "=" * 60)
    print("[OK] DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the generated JSONL files")
    print("2. Run the fine-tuning script: python scripts/training/finetune_gpt4o_mini.py")
    print("3. Wait for fine-tuning to complete (can take hours)")
    print("4. Run comparison evaluation with all models")


if __name__ == "__main__":
    main()


