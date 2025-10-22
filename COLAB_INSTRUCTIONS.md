# Google Colab Fine-Tuning Instructions

## Quick Start (5 Steps)

### 1. Open Google Colab
- Go to: https://colab.research.google.com/
- Click "New Notebook"

### 2. Enable GPU
- Click `Runtime` → `Change runtime type`
- Select `T4 GPU` (free) or `L4 GPU` (if available)
- Click `Save`

### 3. Install Dependencies (Cell 1)
```python
!pip install -q transformers accelerate peft pillow torch torchvision trl datasets
print("✅ Dependencies installed!")
```

### 4. Upload Files (Cell 2)
```python
from google.colab import files
import os

# Upload training data
print("📁 Upload your vlm_safety_training_data.json file:")
uploaded = files.upload()
data_file = list(uploaded.keys())[0]
print(f"✅ Uploaded: {data_file}")

# Create image directory
os.makedirs("data/samples/Sample SVI", exist_ok=True)
print("\n🖼️  Now upload your images:")
print("1. Click the folder icon on the left sidebar")
print("2. Navigate to: data/samples/Sample SVI")
print("3. Click the upload button and select all your .jpg images")
print("4. Wait for upload to complete, then continue...")
```

### 5. Train Model (Cell 3)
```python
import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# Dataset class
class SupervisedDataset(Dataset):
    def __init__(self, data_path, processor):
        with open(data_path, "r", encoding="utf-8") as f:
            self.list_data_dict = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        data = self.list_data_dict[i]
        filename = Path(data['image_path']).name
        image_path = f"data/samples/Sample SVI/{filename}"
        image = Image.open(image_path).convert("RGB")
        text = f"<|image_pad|>{data['prompt']}"
        return {"text": text, "image": image}

# Collate function
def collate_fn(batch, processor):
    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

# Training function
def train_model():
    print("🚀 Loading model...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True, torch_dtype=torch.float16
    )
    
    print("⚙️  Applying LoRA...")
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    device = torch.device("cuda")
    model = model.to(device)
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    
    print("📊 Loading dataset...")
    dataset = SupervisedDataset(data_file, processor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True,
                           collate_fn=lambda b: collate_fn(b, processor))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    print(f"🎯 Training for 3 epochs...")
    model.train()
    
    for epoch in range(3):
        total_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/3")
        
        for batch in progress:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
    
    print("💾 Saving model...")
    model.save_pretrained("fine_tuned_qwen2vl")
    processor.save_pretrained("fine_tuned_qwen2vl")
    print("✅ Training complete! Model saved to 'fine_tuned_qwen2vl'")

# Start training
train_model()
```

### 6. Download Model (Cell 4)
```python
# Zip the model
!zip -r fine_tuned_model.zip fine_tuned_qwen2vl

# Download
from google.colab import files
files.download('fine_tuned_model.zip')
```

---

## What Files to Upload?

1. **Training Data JSON** (`configs/vlm_safety_training_data.json`)
   - Upload this in Cell 2

2. **Images** (30 JPG files from `data/samples/Sample SVI/`)
   - Upload these via the Files panel after running Cell 2

---

## Expected Training Time

- **With T4 GPU**: ~30-45 minutes for 3 epochs
- **With L4 GPU**: ~15-20 minutes for 3 epochs

---

## Tips

- **Free GPU limits**: Colab gives you limited GPU time. If disconnected, you'll need to restart.
- **Keep the browser tab open**: Colab will disconnect if inactive for too long.
- **Download frequently**: Download checkpoints after each epoch to avoid losing progress.

---

## Troubleshooting

**"Runtime disconnected"**
- Colab has usage limits. Try again later or upgrade to Colab Pro.

**"Out of memory"**
- Reduce batch_size from 2 to 1 in the training cell.

**"Images not found"**
- Make sure images are uploaded to exactly: `data/samples/Sample SVI/`

