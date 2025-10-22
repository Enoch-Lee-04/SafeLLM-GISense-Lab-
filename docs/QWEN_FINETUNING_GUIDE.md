# Qwen VLM Fine-tuning for Street View Safety Assessment

This guide will help you fine-tune a Qwen Vision-Language Model (VLM) for street view safety assessment using your annotated data.

## Overview

The setup includes:
- **Qwen2-VL-2B-Instruct** as the base model
- Fine-tuning on your street view images with safety annotations
- Multiple prompt types for different safety assessment tasks
- Comprehensive training and inference scripts

## Quick Start

### 1. Automated Setup
```bash
python setup_qwen_training.py
```

This will:
- Check system requirements
- Install dependencies
- Convert your data to Qwen format
- Create training and inference scripts

### 2. Start Training
```bash
python train_qwen.py
```

### 3. Run Inference
```bash
python run_inference.py "Sample SVI/1.jpg" safety_score
```

## Manual Setup

### 1. Install Dependencies
```bash
pip install -r qwen_requirements.txt
```

### 2. Convert Data
```bash
python convert_data_for_qwen.py
```

### 3. Train Model
```bash
python qwen_safety_finetune.py \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --data_path qwen_training_data.json \
    --output_dir ./qwen_safety_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5
```

### 4. Run Inference
```bash
python qwen_safety_inference.py \
    --model_path ./qwen_safety_model \
    --image_path "Sample SVI/1.jpg" \
    --prompt_type safety_score
```

## Data Format

Your training data is automatically converted to Qwen's conversation format:

```json
{
  "id": "safety_example_1",
  "image_path": "Sample SVI/1.jpg",
  "prompt": "Analyze this street view image...",
  "expected_response": "Safety Score: 7\nReasoning: ...",
  "task_type": "safety_score",
  "category": "safety_scoring",
  "conversation": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Analyze this street view image..."}
      ]
    },
    {
      "role": "assistant",
      "content": "Safety Score: 7\nReasoning: ..."
    }
  ]
}
```

## Training Configuration

### Model Arguments
- **model_name_or_path**: `Qwen/Qwen2-VL-2B-Instruct` (2B parameter model)
- **trust_remote_code**: `True` (required for Qwen models)
- **use_cache**: `False` (for training)

### Data Arguments
- **data_path**: Path to converted training data
- **image_folder**: Folder containing street view images
- **max_length**: Maximum sequence length (2048)
- **train_split**: Training/validation split ratio (0.8)

### Training Arguments
- **num_train_epochs**: 3 (adjust based on data size)
- **per_device_train_batch_size**: 2 (adjust based on GPU memory)
- **gradient_accumulation_steps**: 4 (effective batch size = 8)
- **learning_rate**: 5e-5 (standard for fine-tuning)
- **warmup_ratio**: 0.1 (10% of training steps)

## Prompt Types

### 1. Safety Score (1-10)
```
Analyze this street view image and provide a safety score from 1-10 (10 being safest).

Consider:
- Sidewalk condition and width
- Traffic density and speed
- Street lighting
- Road maintenance
- Pedestrian-vehicle separation
- Visibility and sightlines

Format your response as:
Safety Score: [1-10]
Reasoning: [Brief explanation]
```

### 2. Binary Classification (SAFE/UNSAFE)
```
Classify this street view image as SAFE or UNSAFE for pedestrians.

SAFE = Good sidewalks, low traffic, good lighting, well-maintained infrastructure
UNSAFE = Poor sidewalks, heavy traffic, poor lighting, infrastructure issues

Response format:
Classification: [SAFE/UNSAFE]
Confidence: [HIGH/MEDIUM/LOW]
Reason: [Brief explanation]
```

### 3. Detailed Analysis
```
Provide a detailed safety analysis of this street view image.

Evaluate these aspects (1-10 scale each):
1. Pedestrian Safety: ___
2. Traffic Safety: ___
3. Lighting Safety: ___
4. Infrastructure Safety: ___
5. Crime Safety: ___

Overall Score: ___/50
Main Concerns: [List top 3 issues]
Strengths: [List top 3 positive features]
```

### 4. Risk Assessment
```
Assess the safety risks in this street view image.

Identify risks by category:
HIGH RISK: [List high-risk elements]
MEDIUM RISK: [List medium-risk elements]
LOW RISK: [List low-risk elements]

Overall Risk Level: [LOW/MEDIUM/HIGH]
Primary Risk Factor: [Main safety concern]
```

## Hardware Requirements

### Minimum Requirements
- **CPU**: Multi-core processor (training will be slow)
- **RAM**: 16GB
- **Storage**: 20GB free space

### Recommended Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/4070 or better)
- **RAM**: 32GB
- **Storage**: 50GB free space

### GPU Memory Usage
- **Batch size 1**: ~6GB VRAM
- **Batch size 2**: ~10GB VRAM
- **Batch size 4**: ~18GB VRAM

## Training Tips

### 1. Start Small
- Begin with fewer epochs (1-2)
- Use smaller batch sizes if you encounter memory issues
- Monitor training loss and validation metrics

### 2. Data Quality
- Ensure image paths are correct
- Verify annotation quality
- Balance different prompt types in your dataset

### 3. Hyperparameter Tuning
- **Learning rate**: Start with 5e-5, try 1e-5 or 1e-4
- **Batch size**: Increase if you have more GPU memory
- **Epochs**: More data may require fewer epochs

### 4. Monitoring
- Use Weights & Biases (wandb) for experiment tracking
- Monitor both training and validation loss
- Check for overfitting

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
--per_device_train_batch_size 1

# Increase gradient accumulation
--gradient_accumulation_steps 8
```

#### 2. Image Loading Errors
- Check image paths in your data
- Ensure images are in supported formats (JPG, PNG)
- Verify file permissions

#### 3. Model Loading Issues
- Ensure you have internet connection for initial model download
- Check if you have sufficient disk space
- Verify transformers library version

#### 4. Training Convergence
- Try different learning rates
- Increase warmup ratio
- Check data quality and balance

### Performance Optimization

#### 1. Use Mixed Precision
```python
# Automatically enabled when CUDA is available
--fp16 True
```

#### 2. Gradient Checkpointing
```python
# Add to model loading
model.gradient_checkpointing_enable()
```

#### 3. Data Loading
```python
# Use multiple workers
--dataloader_num_workers 4
```

## Evaluation

### Metrics to Monitor
- **Training Loss**: Should decrease over time
- **Validation Loss**: Should track training loss
- **BLEU Score**: For response quality (optional)
- **Accuracy**: For binary classification tasks

### Sample Evaluation
```python
# After training, test on sample images
python qwen_safety_inference.py \
    --model_path ./qwen_safety_model \
    --image_folder "Sample SVI" \
    --prompt_type safety_score \
    --output_file evaluation_results.json
```

## Advanced Usage

### Custom Prompts
```python
# Use custom prompts for specific assessments
python qwen_safety_inference.py \
    --model_path ./qwen_safety_model \
    --image_path "image.jpg" \
    --custom_prompt "Assess pedestrian accessibility in this image..."
```

### Batch Processing
```python
# Process multiple images
python qwen_safety_inference.py \
    --model_path ./qwen_safety_model \
    --image_folder "test_images" \
    --prompt_type detailed_analysis \
    --output_file batch_results.json
```

### Model Comparison
```python
# Compare different model checkpoints
python qwen_safety_inference.py \
    --model_path ./qwen_safety_model/checkpoint-1000 \
    --image_path "test.jpg" \
    --prompt_type safety_score
```

## Next Steps

After successful training:

1. **Validate Model**: Test on held-out images
2. **Deploy Model**: Create API or web interface
3. **Scale Up**: Train on larger datasets
4. **Fine-tune Further**: Adjust for specific use cases
5. **Compare Models**: Evaluate against other VLMs

## Resources

- [Qwen-VL Paper](https://arxiv.org/abs/2308.12966)
- [Qwen GitHub](https://github.com/QwenLM/Qwen-VL)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Weights & Biases](https://wandb.ai/)

## Support

If you encounter issues:
1. Check this guide's troubleshooting section
2. Review the training logs
3. Verify your data format
4. Check system requirements

Good luck with your Qwen VLM fine-tuning for safety assessment!
