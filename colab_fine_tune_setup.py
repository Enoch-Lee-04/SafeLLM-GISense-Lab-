"""
Google Colab Setup Script for Qwen2-VL Fine-tuning
===================================================

Instructions:
1. Go to https://colab.research.google.com/
2. Create a new notebook
3. Go to Runtime > Change runtime type > Select "T4 GPU" or "L4 GPU"
4. Copy this entire script into the first cell and run it
5. Upload your training data JSON file when prompted
"""

# ========== STEP 1: Install Dependencies ==========
print("📦 Installing dependencies...")
!pip install -q transformers accelerate peft pillow torch torchvision trl datasets

# ========== STEP 2: Upload Training Data ==========
print("\n📁 Upload your training data file (vlm_safety_training_data.json)...")
from google.colab import files
import json
import os

uploaded = files.upload()
data_filename = list(uploaded.keys())[0]
print(f"✅ Uploaded: {data_filename}")

# ========== STEP 3: Download and Prepare Images ==========
print("\n🖼️  Setting up image directory...")
# Create images directory
os.makedirs("data/samples/Sample SVI", exist_ok=True)

# Read the JSON to get image paths
with open(data_filename, 'r') as f:
    training_data = json.load(f)

print(f"📊 Found {len(training_data)} training samples")
print("\n⚠️  IMPORTANT: Now upload your images to 'data/samples/Sample SVI' directory")
print("Click the folder icon on the left, navigate to 'data/samples/Sample SVI', and upload all your .jpg files")
print("\nWaiting for you to upload images... (This script will continue once you're done)")

input("Press Enter after you've uploaded all images...")

# Verify images are present
from pathlib import Path
image_files = list(Path("data/samples/Sample SVI").glob("*.jpg"))
print(f"✅ Found {len(image_files)} images")

# ========== STEP 4: Training Script ==========
print("\n🚀 Creating training script...")

training_script = '''
import os
import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, processor: AutoProcessor):
        super(SupervisedDataset, self).__init__()
        with open(data_path, "r", encoding="utf-8") as f:
            self.list_data_dict = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        data = self.list_data_dict[i]
        filename = Path(data['image_path']).name
        image_path = os.path.join("data", "samples", "Sample SVI", filename)
        image = Image.open(image_path).convert("RGB")
        text_with_image = f"<|image_pad|>{data['prompt']}"
        return {"text": text_with_image, "image": image}

def collate_fn(batch, processor):
    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

def train(data_path, output_dir, num_epochs=3, batch_size=1, learning_rate=5e-5):
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", 
        trust_remote_code=True, 
        torch_dtype=torch.float16
    )

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    print("Loading dataset...")
    train_dataset = SupervisedDataset(data_path=data_path, processor=processor)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Starting training for {num_epochs} epochs...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved to {checkpoint_dir}")

    print("Saving final model...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("Training complete!")

if __name__ == "__main__":
    train(
        data_path="DATA_PATH_PLACEHOLDER",
        output_dir="fine_tuned_qwen2vl",
        num_epochs=3,
        batch_size=1,
        learning_rate=5e-5
    )
'''

# Save training script with actual data path
training_script = training_script.replace("DATA_PATH_PLACEHOLDER", data_filename)
with open("train.py", "w") as f:
    f.write(training_script)

print("✅ Training script created: train.py")

# ========== STEP 5: Start Training ==========
print("\n🎯 Starting training...")
print("="*50)
!python train.py

# ========== STEP 6: Download Results ==========
print("\n📥 Downloading fine-tuned model...")
print("The model will be saved in 'fine_tuned_qwen2vl' directory")
print("You can download it using the Files panel on the left, or run:")
print("!zip -r fine_tuned_model.zip fine_tuned_qwen2vl")
print("Then download the zip file")

