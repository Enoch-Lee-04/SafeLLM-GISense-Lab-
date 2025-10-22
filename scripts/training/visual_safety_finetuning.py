#!/usr/bin/env python3
"""
Visual Safety Assessment Fine-tuning Pipeline
Fine-tune Qwen-VL for street view image safety evaluation with structured data
"""

import os
import json
import torch
import argparse
import warnings
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    HfArgumentParser
)
from transformers.trainer_utils import get_last_checkpoint
import wandb
from tqdm import tqdm


# Disable some warnings
logging.getLogger("transformers").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


@dataclass
class ModelArguments:
    """Model configuration arguments"""
    model_name_or_path: str = field(
        default="Qwen/Qwen2-VL-2B-Instruct",
        metadata={"help": "Path to pretrained model"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code"}
    )
    use_cache: bool = field(
        default=False,
        metadata={"help": "Whether to use cache"}
    )


@dataclass
class DataArguments:
    """Data configuration arguments"""
    data_path: str = field(
        default="./vlm_safety_training_data.json",
        metadata={"help": "Path to training data"}
    )
    validation_split: float = field(
        default=0.2,
        metadata={"help": "Fraction of data for validation"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )


@dataclass
class SafetyTrainingArguments:
    """Training configuration for safety assessment"""
    output_dir: str = field(
        default="./visual_safety_model",
        metadata={"help": "Output directory for model"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Training batch size per device"}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Evaluation batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=8,
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
        default=10,
        metadata={"help": "Logging steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint steps"}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "Evaluation steps"}
    )
    save_total_limit: int = field(
        default=3,
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
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use fp16"}
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of dataloader workers"}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Remove unused columns"}
    )


class SafetyDataset(Dataset):
    """Dataset for visual safety assessment training"""
    
    def __init__(self, data: List[Dict], processor, tokenizer, max_length: int = 512):
        self.data = data
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = item['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a placeholder or skip
            return None
        
        # Get prompt and expected response
        prompt = item['prompt']
        expected_response = item['expected_response']
        
        # Create conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            },
            {
                "role": "assistant",
                "content": expected_response
            }
        ]
        
        try:
            # Process conversation
            processed = self.processor(
                conversation=conversation,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # Extract input_ids and attention_mask
            input_ids = processed['input_ids'].squeeze(0)
            attention_mask = processed['attention_mask'].squeeze(0)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'image_id': Path(image_path).stem
            }
        
        except Exception as e:
            print(f"Error processing conversation for {image_path}: {e}")
            return None


class SafetyTrainer(Trainer):
    """Custom trainer for safety assessment fine-tuning"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        # Use standard language modeling loss
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs.get('loss')
        
        return (loss, outputs) if return_outputs else loss


def prepare_datasets(
    data_path: str, 
    validation_split: float, 
    processor, 
    tokenizer, 
    max_length: int
) -> tuple[SafetyDataset, SafetyDataset]:
    """Prepare training and validation datasets"""
    
    # Load training data
    with open(data_path, 'r') as f:
        full_data = json.load(f)
    
    # Shuffle data
    np.random.seed(42)
    indices = np.random.permutation(len(full_data))
    full_data = [full_data[i] for i in indices]
    
    # Split into train/validation
    split_idx = int(len(full_data) * (1 - validation_split))
    train_data = full_data[:split_idx]
    val_data = full_data[split_idx:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create datasets
    train_dataset = SafetyDataset(train_data, processor, tokenizer, max_length)
    val_dataset = SafetyDataset(val_data, processor, tokenizer, max_length)
    
    return train_dataset, val_dataset


def collate_fn(batch):
    """Custom collate function for handling None values"""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return None
    
    # Pad sequences
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    
    # Find max length in batch
    max_length = max(len(ids) for ids in input_ids)
    
    # Pad sequences
    padded_input_ids = []
    padded_attention_mask = []
    
    for ids, mask in zip(input_ids, attention_mask):
        pad_length = max_length - len(ids)
        
        padded_ids = torch.cat([ids, torch.full((pad_length,), tokenizer.pad_token_id)])
        padded_mask = torch.cat([mask, torch.zeros(pad_length)])
        
        padded_input_ids.append(padded_ids)
        padded_attention_mask.append(padded_mask)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_mask),
        'labels': torch.stack(padded_input_ids)  # For language modeling loss
    }


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, SafetyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Initialize wandb
    if training_args.output_dir and os.path.exists(training_args.output_dir):
        wandb.init(
            project="visual-safety-assessment",
            name=f"safety-training-{training_args.output_dir}",
            config={
                "model": model_args.model_name_or_path,
                "epochs": training_args.num_train_epochs,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
            }
        )
    
    print(f"Loading model: {model_args.model_name_or_path}")
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float16 if training_args.fp16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Disable cache for training
    if not model_args.use_cache:
        model.config.use_cache = False
    
    print("Loading datasets...")
    train_dataset, val_dataset = prepare_datasets(
        data_args.data_path,
        data_args.validation_split,
        processor,
        tokenizer,
        data_args.max_length
    )
    
    # Initialize trainer
    trainer = SafetyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )
    
    # Training
    print("Starting training...")
    
    # Check for resume
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir):
        checkpoint = get_last_checkpoint(training_args.output_dir)
    
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save model
    trainer.save_model()
    
    # Evaluate final model
    print("Evaluating final model...")
    eval_results = trainer.evaluate()
    
    print(f"Training completed! Final evaluation results: {eval_results}")
    
    # Save training results
    results_path = os.path.join(training_args.output_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"Training results saved to {results_path}")


if __name__ == "__main__":
    main()
