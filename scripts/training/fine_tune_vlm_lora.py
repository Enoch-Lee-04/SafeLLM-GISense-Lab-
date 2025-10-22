#!/usr/bin/env python3
"""
Fine-tunes a Qwen-VL model using PEFT/LoRA with a custom training loop.
"""
import os
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import argparse

# --- Dataset Class ---
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, processor: AutoProcessor):
        super(SupervisedDataset, self).__init__()
        with open(data_path, "r", encoding="utf-8") as f:
            self.list_data_dict = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        data = self.list_data_dict[i]

        # Extract filename and use the correct project-relative path
        filename = Path(data['image_path']).name
        image_path = os.path.join("data", "samples", "Sample SVI", filename)
        image = Image.open(image_path).convert("RGB")

        # Create text with image pad token
        text_with_image = f"<|image_pad|>{data['prompt']}"

        return {
            "text": text_with_image,
            "image": image,
        }

def collate_fn(batch, processor):
    """Collate function to batch samples."""
    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    
    # Process batch with the processor
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    # Set labels to input_ids for language modeling
    inputs["labels"] = inputs["input_ids"].clone()
    
    return inputs

# --- Main Training ---
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--fp16", type=bool, default=False)
    args = parser.parse_args()

    # Load model and processor
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", 
        trust_remote_code=True, 
        torch_dtype=torch.float16 if args.fp16 else torch.float32
    )

    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Move model to device (supports CUDA, DirectML for AMD, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        try:
            import torch_directml
            device = torch_directml.device()
            print(f"Using AMD GPU (DirectML): {device}")
        except ImportError:
            device = torch.device("cpu")
            print("Using CPU (no GPU acceleration)")
    
    model = model.to(device)

    # Create dataset and dataloader
    print("Loading dataset...")
    train_dataset = SupervisedDataset(data_path=args.data_path, processor=processor)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor)
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Training loop
    print(f"Starting training for {args.num_train_epochs} epochs...")
    model.train()
    
    for epoch in range(args.num_train_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass with gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            # Update weights
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * args.gradient_accumulation_steps
            progress_bar.set_postfix({"loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}"})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint after each epoch
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved to {checkpoint_dir}")

    # Save final model
    print(f"Saving final model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Training complete!")

if __name__ == "__main__":
    train()
