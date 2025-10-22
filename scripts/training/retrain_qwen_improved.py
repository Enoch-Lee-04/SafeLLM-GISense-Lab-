"""
Improved Qwen VLM Fine-tuning with Better Training Configuration
Addresses repetition issues and improves model performance
"""

import os
import json
import torch
import warnings
from typing import Dict, List
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    HfArgumentParser
)

warnings.filterwarnings("ignore")


@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: str = field(
        default="Qwen/Qwen2-VL-2B-Instruct",
        metadata={"help": "Path to pretrained model"}
    )
    trust_remote_code: bool = field(default=True)


@dataclass
class DataArguments:
    """Arguments for data configuration"""
    data_path: str = field(
        default="configs/vlm_safety_training_data.json",
        metadata={"help": "Path to training data JSON"}
    )
    image_folder: str = field(
        default="data/images/image",
        metadata={"help": "Folder containing images"}
    )
    max_length: int = field(default=512)


@dataclass
class CustomTrainingArguments:
    """Arguments for training"""
    output_dir: str = field(
        default="models/fine_tuned_qwen2vl_improved",
        metadata={"help": "Output directory"}
    )
    num_train_epochs: int = field(default=5)  # Increased from 1-2
    per_device_train_batch_size: int = field(default=2)  # Increased if memory allows
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-5)  # Slightly lower for stability
    warmup_steps: int = field(default=50)  # More warmup
    logging_steps: int = field(default=5)
    save_steps: int = field(default=50)
    save_total_limit: int = field(default=3)


class ImprovedSafetyDataset(Dataset):
    """Improved dataset with better prompt formatting"""
    
    def __init__(
        self, 
        data: List[Dict], 
        processor,
        image_folder: str,
        max_length: int = 512
    ):
        self.data = data
        self.processor = processor
        self.image_folder = image_folder
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_name = os.path.basename(item['image_path'])
        image_path = os.path.join(self.image_folder, image_name)
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Create improved conversation format
            # Use a consistent, clear format that the model can learn
            conversation_text = f"<|image_pad|>User: {item['prompt']}\nAssistant: {item['expected_response']}<|endoftext|>"
            
            # Process with image
            processed = self.processor(
                text=[conversation_text],
                images=[image],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding="max_length"  # Ensure consistent length
            )
            
            pixel_values = processed.get('pixel_values')
            if pixel_values is not None:
                pixel_values = pixel_values.squeeze(0)
            
            image_grid_thw = processed.get('image_grid_thw')
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.squeeze(0)
                
        except Exception as e:
            print(f"Warning: Error processing {image_path}: {e}")
            # Fallback to text-only with dummy image
            conversation_text = f"<|image_pad|>User: {item['prompt']}\nAssistant: {item['expected_response']}<|endoftext|>"
            processed = self.processor(
                text=[conversation_text],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )
            pixel_values = None
            image_grid_thw = None
        
        input_ids = processed['input_ids'].squeeze(0)
        attention_mask = processed['attention_mask'].squeeze(0)
        
        # Create labels for training
        labels = input_ids.clone()
        
        # Mask the user prompt part (only train on assistant response)
        # Find where "Assistant:" starts
        tokenized_prompt = self.processor.tokenizer.encode(
            f"<|image_pad|>User: {item['prompt']}\nAssistant: ",
            add_special_tokens=False
        )
        prompt_length = len(tokenized_prompt)
        
        # Mask everything before the assistant's response
        labels[:prompt_length] = -100
        
        # Also mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        if pixel_values is not None:
            result['pixel_values'] = pixel_values
        if image_grid_thw is not None:
            result['image_grid_thw'] = image_grid_thw
            
        return result


def create_improved_collate_fn(processor):
    """Create collate function with proper handling"""
    
    def collate_fn(batch: List[Dict]) -> Dict:
        from torch.nn.utils.rnn import pad_sequence
        
        batch = [item for item in batch if item is not None]
        if not batch:
            return {}
        
        pad_token_id = processor.tokenizer.pad_token_id or 151643
        
        # Pad sequences
        input_ids = pad_sequence(
            [item['input_ids'] for item in batch],
            batch_first=True,
            padding_value=pad_token_id
        )
        attention_mask = pad_sequence(
            [item['attention_mask'] for item in batch],
            batch_first=True,
            padding_value=0
        )
        labels = pad_sequence(
            [item['labels'] for item in batch],
            batch_first=True,
            padding_value=-100
        )
        
        collated = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        # Handle pixel_values
        pixel_values_list = [item.get('pixel_values') for item in batch]
        if any(pv is not None for pv in pixel_values_list):
            first_valid = next(pv for pv in pixel_values_list if pv is not None)
            shape, dtype = first_valid.shape, first_valid.dtype
            
            processed = [
                pv if pv is not None else torch.zeros(shape, dtype=dtype)
                for pv in pixel_values_list
            ]
            collated['pixel_values'] = torch.stack(processed)
        
        # Handle image_grid_thw
        grid_list = [item.get('image_grid_thw') for item in batch]
        if any(gt is not None for gt in grid_list):
            first_valid = next(gt for gt in grid_list if gt is not None)
            shape, dtype = first_valid.shape, first_valid.dtype
            
            processed = [
                gt if gt is not None else torch.zeros(shape, dtype=dtype)
                for gt in grid_list
            ]
            collated['image_grid_thw'] = torch.stack(processed)
        
        return collated
    
    return collate_fn


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args_custom = parser.parse_args_into_dataclasses()
    
    print("="*70)
    print("Improved Qwen2-VL Fine-tuning for Street View Safety")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {model_args.model_name_or_path}")
    print(f"  Data: {data_args.data_path}")
    print(f"  Images: {data_args.image_folder}")
    print(f"  Output: {training_args_custom.output_dir}")
    print(f"  Epochs: {training_args_custom.num_train_epochs}")
    print(f"  Batch size: {training_args_custom.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args_custom.learning_rate}")
    
    # Load data
    print(f"\nLoading training data from {data_args.data_path}...")
    with open(data_args.data_path, 'r') as f:
        training_data = json.load(f)
    
    print(f"Loaded {len(training_data)} training examples")
    
    # Load processor and model
    print(f"\nLoading model: {model_args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float32
    )
    
    print("Model loaded successfully!")
    
    # Create dataset
    print("\nPreparing dataset...")
    train_dataset = ImprovedSafetyDataset(
        training_data,
        processor,
        data_args.image_folder,
        data_args.max_length
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    
    # Create collate function
    collate_fn = create_improved_collate_fn(processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=training_args_custom.output_dir,
        num_train_epochs=training_args_custom.num_train_epochs,
        per_device_train_batch_size=training_args_custom.per_device_train_batch_size,
        gradient_accumulation_steps=training_args_custom.gradient_accumulation_steps,
        learning_rate=training_args_custom.learning_rate,
        warmup_steps=training_args_custom.warmup_steps,
        logging_steps=training_args_custom.logging_steps,
        save_steps=training_args_custom.save_steps,
        save_total_limit=training_args_custom.save_total_limit,
        remove_unused_columns=False,
        label_names=["labels"],
        logging_dir=f"{training_args_custom.output_dir}/logs",
        report_to="none",  # Disable wandb
        save_strategy="steps",
        fp16=False,  # Use fp32 for stability
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )
    
    # Train
    print("\n" + "="*70)
    print("Starting improved training...")
    print("="*70 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(training_args_custom.output_dir)
    processor.save_pretrained(training_args_custom.output_dir)
    
    print("\n" + "="*70)
    print(f"Training complete! Model saved to: {training_args_custom.output_dir}")
    print("="*70)
    
    print("\nNext steps:")
    print(f"  1. Test the model: python scripts/evaluation/test_fine_tuned_qwen.py --model_path {training_args_custom.output_dir}")
    print(f"  2. Check results in the generated Excel file")


if __name__ == "__main__":
    main()

