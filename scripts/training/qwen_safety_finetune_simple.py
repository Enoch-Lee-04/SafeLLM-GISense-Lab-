"""
Simplified Qwen VLM Fine-tuning for Street View Safety Assessment
Works without DeepSpeed and other optional dependencies
"""

import os
import sys
import json
import torch
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    HfArgumentParser
)
from tqdm import tqdm


@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: str = field(
        default="Qwen/Qwen2-VL-2B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading model"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration"""
    data_path: str = field(
        default="qwen_training_data.json",
        metadata={"help": "Path to training data JSON file"}
    )
    image_folder: str = field(
        default="Sample SVI",
        metadata={"help": "Folder containing street view images"}
    )
    max_length: int = field(
        default=1024,  # Reduced for CPU training
        metadata={"help": "Maximum sequence length"}
    )
    train_split: float = field(
        default=0.8,
        metadata={"help": "Fraction of data to use for training"}
    )


@dataclass
class CustomTrainingArguments:
    """Arguments for training configuration"""
    output_dir: str = field(
        default="./qwen_safety_model",
        metadata={"help": "Output directory for model"}
    )
    num_train_epochs: int = field(
        default=2,  # Reduced for testing
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=1,  # CPU-friendly batch size
        metadata={"help": "Training batch size per device"}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Evaluation batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=8,  # Increased to maintain effective batch size
        metadata={"help": "Number of steps to accumulate gradients"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Learning rate"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Warmup ratio"}
    )
    logging_steps: int = field(
        default=5,
        metadata={"help": "Logging steps"}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save checkpoint steps"}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Evaluation steps"}
    )
    save_total_limit: int = field(
        default=2,
        metadata={"help": "Maximum number of checkpoints to save"}
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Load best model at end"}
    )
    metric_for_best_model: str = field(
        default="eval_loss",
        metadata={"help": "Metric for best model selection"}
    )
    greater_is_better: bool = field(
        default=False,
        metadata={"help": "Whether greater metric is better"}
    )
    report_to: str = field(
        default="none",  # Disable wandb for simplicity
        metadata={"help": "Reporting tool (wandb, tensorboard, none)"}
    )
    run_name: str = field(
        default="qwen_safety_assessment",
        metadata={"help": "Run name for logging"}
    )


class SafetyDataset(Dataset):
    """Dataset for safety assessment fine-tuning"""
    
    def __init__(
        self, 
        data: List[Dict], 
        processor: AutoProcessor,
        image_folder: str,
        max_length: int = 1024,
        is_training: bool = True
    ):
        self.data = data
        self.processor = processor
        self.image_folder = image_folder
        self.max_length = max_length
        self.is_training = is_training
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image from the correct folder
        image_name = os.path.basename(item['image_path'])
        image_path = os.path.join(self.image_folder, image_name)
        
        pixel_values = None
        image_grid_thw = None
        try:
            image = Image.open(image_path).convert('RGB')
            # Process image and text together with proper image placeholder
            text_with_image = f"<|image_pad|>User: {item['prompt']}\nAssistant: {item['expected_response']}"
            processed = self.processor(
                text=[text_with_image],
                images=[image],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            )
            if 'pixel_values' in processed and processed['pixel_values'] is not None:
                pixel_values = processed['pixel_values'].squeeze(0)
            if 'image_grid_thw' in processed and processed['image_grid_thw'] is not None:
                image_grid_thw = processed['image_grid_thw'].squeeze(0)

        except Exception as e:
            print(f"Warning: Could not process image {image_path}: {e}. Proceeding with text only.")
            # Fallback to text-only processing with image placeholder for consistency
            text_without_image = f"<|image_pad|>User: {item['prompt']}\nAssistant: {item['expected_response']}"
            processed = self.processor(
                text=[text_without_image],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            )

        input_ids = processed['input_ids'].squeeze(0)
        labels = input_ids.clone()

        # Simple masking for the prompt part
        # A more robust solution would find the start of the assistant's response.
        mask_length = int(len(input_ids) * 0.6) # Approximate prompt length
        labels[:mask_length] = -100

        result = {
            'input_ids': input_ids,
            'attention_mask': processed['attention_mask'].squeeze(0),
            'labels': labels,
            'pixel_values': pixel_values
        }

        # Add image_grid_thw if present
        if image_grid_thw is not None:
            result['image_grid_thw'] = image_grid_thw

        return result


def load_training_data(data_path: str) -> List[Dict]:
    """Load and preprocess training data"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} training examples")
    return data


def split_data(data: List[Dict], train_split: float = 0.8) -> tuple:
    """Split data into train and validation sets"""
    random.shuffle(data)
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    return train_data, val_data


def main():
    print("Qwen VLM Safety Assessment Fine-tuning (Simplified)")
    print("=" * 60)
    
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, custom_args = parser.parse_args_into_dataclasses()
    
    print("Loading model and processor...")
    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch.float32,
            device_map=None,
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Define a robust collate_fn that has access to the processor
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        from torch.nn.utils.rnn import pad_sequence

        batch = [item for item in batch if item is not None and 'input_ids' in item]
        if not batch:
            return {}

        pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 151643

        input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=pad_token_id)
        attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)

        collated_batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

        # Handle pixel_values
        pixel_values_list = [item.get('pixel_values') for item in batch]
        if any(pv is not None for pv in pixel_values_list):
            # Use shape from first valid pixel_values
            first_valid_pv = next(pv for pv in pixel_values_list if pv is not None)
            shape, dtype = first_valid_pv.shape, first_valid_pv.dtype

            processed_pixel_values = [
                pv if pv is not None else torch.zeros(shape, dtype=dtype)
                for pv in pixel_values_list
            ]
            collated_batch['pixel_values'] = torch.stack(processed_pixel_values)
        else:
            # Default shape for Qwen2-VL: (num_patches, patch_dim)
            shape = (len(batch), 784, 1176)
            collated_batch['pixel_values'] = torch.zeros(shape, dtype=torch.float32)

        # Handle image_grid_thw if present
        grid_thw_list = [item.get('image_grid_thw') for item in batch]
        if any(gt is not None for gt in grid_thw_list):
            first_valid_gt = next(gt for gt in grid_thw_list if gt is not None)
            shape, dtype = first_valid_gt.shape, first_valid_gt.dtype

            processed_grid_thw = [
                gt if gt is not None else torch.zeros(shape, dtype=dtype)
                for gt in grid_thw_list
            ]
            collated_batch['image_grid_thw'] = torch.stack(processed_grid_thw)
        else:
            # Default for Qwen2-VL when no images: (1, 1, 1) per sample
            shape = (len(batch), 1, 1)
            collated_batch['image_grid_thw'] = torch.zeros(shape, dtype=torch.long)

        return collated_batch

    print("Loading training data...")
    if not os.path.exists(data_args.data_path):
        print(f"Error: Training data file {data_args.data_path} not found!")
        return 1
    
    data = load_training_data(data_args.data_path)
    
    if len(data) > 20:
        print(f"Using subset of {min(20, len(data))} examples for testing")
        data = data[:20]
    
    train_data, val_data = split_data(data, data_args.train_split)
    
    train_dataset = SafetyDataset(
        train_data, 
        processor, 
        data_args.image_folder,
        data_args.max_length, 
        is_training=True
    )
    val_dataset = SafetyDataset(
        val_data, 
        processor, 
        data_args.image_folder,
        data_args.max_length, 
        is_training=False
    )
    
    training_args = TrainingArguments(
        output_dir=custom_args.output_dir,
        num_train_epochs=custom_args.num_train_epochs,
        per_device_train_batch_size=custom_args.per_device_train_batch_size,
        per_device_eval_batch_size=custom_args.per_device_eval_batch_size,
        gradient_accumulation_steps=custom_args.gradient_accumulation_steps,
        learning_rate=custom_args.learning_rate,
        warmup_ratio=custom_args.warmup_ratio,
        logging_steps=custom_args.logging_steps,
        save_steps=custom_args.save_steps,
        eval_steps=custom_args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=custom_args.save_total_limit,
        load_best_model_at_end=custom_args.load_best_model_at_end,
        metric_for_best_model=custom_args.metric_for_best_model,
        greater_is_better=custom_args.greater_is_better,
        report_to=custom_args.report_to,
        run_name=custom_args.run_name,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    if os.path.isdir(training_args.output_dir):
        print(f"Warning: Output directory {training_args.output_dir} already exists. Training will overwrite existing model.")

    print("Starting training...")
    print(f"Training on {len(train_dataset)} examples")
    print(f"Validating on {len(val_dataset)} examples")
    print("Note: CPU training will be slow. Consider using a GPU for faster training.")
    
    try:
        trainer.train()
        
        print("Saving final model...")
        trainer.save_model()
        processor.save_pretrained(training_args.output_dir)
        
        print("Training completed successfully!")
        print(f"Model saved to: {training_args.output_dir}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
