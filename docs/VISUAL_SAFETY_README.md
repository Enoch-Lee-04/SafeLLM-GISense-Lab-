# Visual Safety Assessment with Qwen-VL

A comprehensive system for evaluating street view image safety using Vision-Language Models (VLMs) with pairwise comparison scoring.

## 🎯 Project Overview

This project implements a three-stage pipeline for street view safety assessment:

1. **Baseline Evaluation**: Evaluate street view images using unmodified Qwen-VL model
2. **Anchor-Based Scoring**: Perform pairwise comparisons to generate 0-10 safety scores through relative ranking
3. **Fine-tuning**: Train Qwen-VL on annotated safety data for improved accuracy

## 📁 Project Structure

```
visual-safety-assessment/
├── Core Scripts
│   ├── vlm_baseline_evaluation.py      # Baseline VLM evaluation
│   ├── anchor_based_safety_scoring.py   # Pairwise comparison scoring
│   ├── visual_safety_finetuning.py     # Qwen-VL fine-tuning pipeline
│   ├── evaluation_metrics.py           # Comprehensive evaluation metrics
│   └── qwen_safety_inference.py       # Inference with fine-tuned model
├── Data
│   ├── vlm_safety_training_data.json   # Training annotations
│   └── Sample SVI/                    # Street view images
├── Configuration
│   ├── visual_safety_requirements.txt  # Project dependencies
│   └── VLM_PROMPTS_REFERENCE.md       # Prompt templates
└── Documentation
    ├── VISUAL_SAFETY_README.md        # This file
    ├── DEPLOYMENT_GUIDE.md           # Deployment instructions
    └── QWEN_FINETUNING_GUIDE.md     # Fine-tuning guide
```

## 🚀 Quick Start

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd visual-safety-assessment
pip install -r visual_safety_requirements.txt
```

2. **Download images:**
Place your street view images in the `Sample SVI/` folder.

### Usage

#### Stage 1: Baseline Evaluation

```bash
# Run baseline evaluation with pairwise comparisons
python vlm_baseline_evaluation.py \
    --image_folder "Sample SVI" \
    --training_data vlm_safety_training_data.json \
    --output_dir ./baseline_results \
    --max_comparisons 40
```

#### Stage 2: Anchor-Based Scoring

```bash
# Generate anchor-based safety scores (0-10)
python anchor_based_safety_scoring.py \
    --image_folder "Sample SVI" \
    --output_dir ./anchor_scoring_results \
    --max_comparisons 40 \
    --comparison_strategy balanced
```

#### Stage 3: Model Fine-tuning

```bash
# Fine-tune Qwen-VL on safety assessment data
python visual_safety_finetuning.py \
    --data_path vlm_safety_training_data.json \
    --output_dir ./fine_tuned_model \
    --num_train_epochs 3 \
    --learning_rate 5e-5
```

#### Evaluation and Metrics

```bash
# Generate comprehensive evaluation metrics
python evaluation_metrics.py \
    --training_data vlm_safety_training_data.json \
    --vlm_results ./baseline_results/baseline_pairwise_results.json \
    --output_dir ./evaluation_metrics
```

## 📊 Data Format

### Training Data Structure

The `vlm_safety_training_data.json` contains annotations in this format:

```json
{
    "image_path": "Sample SVI/1.jpg",
    "prompt": "Analyze this street view image and provide a safety score from 1-10...",
    "task_type": "safety_score",
    "expected_format": "Safety Score: [1-10]\nReasoning: [explanation]",
    "expected_response": "Safety Score: 8\nReasoning: well-maintained sidewalks..."
}
```

### Task Types

1. **safety_score**: 0-10 safety scale assessment
2. **binary_classification**: SAFE/UNSAFE classification  
3. **detailed_analysis**: Multi-dimensional safety breakdown
4. **risk_assessment**: Risk level categorization
5. **accessibility**: Accessibility-focused evaluation

## 🔬 Evaluation Methodology

### Anchor-Based Scoring System

The core innovation of this system is anchor-based scoring:

1. **Pairwise Comparisons**: Each image is compared against ~40 other images
2. **Relative Ranking**: Images compete head-to-head for safety assessment
3. **Win Rate Conversion**: Win percentage converted to 0-10 safety score:
   - Images that win almost every comparison → Score ~10
   - Images that rarely win → Score ~0  
   - Middle performers → Graded scale

### Scoring Algorithm

```python
# Sigmoid transformation for score mapping
safety_score = 10 / (1 + exp(-k * (win_rate - 0.5)))
```

Where:
- `k = 6` provides reasonable spread across 0-10 range
- `win_rate` is percentage of comparisons won
- 50% win rate → Score of 5 (neutral)

### Evaluation Metrics

**Regression Metrics (Continuous Scores):**
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)  
- Pearson Correlation Coefficient
- R² Score
- Within-threshold accuracy (≤0.5, ≤1.0, ≤1.5, ≤2.0 points)

**Classification Metrics:**
- Overall Accuracy
- Cohen's Kappa (inter-rater reliability)
- Per-class Precision, Recall, F1-Score
- Confusion Matrix Analysis

## 🛠 Core Components

### 1. Baseline Evaluator (`vlm_baseline_evaluation.py`)

- Uses unmodified Qwen-VL model for unbiased evaluation
- Implements pairwise comparison logic
- Extracts safety scores from model responses
- Compares results with ground truth annotations

### 2. Anchor-Based Scorer (`anchor_based_safety_scoring.py`)

- Generates balanced comparison pairs
- Performs head-to-head safety assessments  
- Tracks win/loss statistics per image
- Converts win rates to standardized 0-10 scores

### 3. Safety Trainer (`visual_safety_finetuning.py`)

- Custom dataset loader for safety assessment data
- Configurable fine-tuning parameters
- Mixed precision training support
- Comprehensive evaluation during training

### 4. Evaluation Metrics (`evaluation_metrics.py`)

- Automated parsing of predictions and ground truth
- Statistical analysis of performance
- Visualization generation (scatter plots, error distributions)
- Cross-validation capabilities

## 📈 Performance Benchmarks

Expected Results:

- **Baseline (Unmodified Qwen-VL)**: 
  - MAE: ~1.5-2.0 points
  - Correlation: ~0.6-0.7
  
- **Fine-tuned Model**:
  - MAE: ~1.0-1.3 points  
  - Correlation: ~0.8-0.85
  - Classification Accuracy: ~85-90%

## 🐳 Docker Support

```bash
# Build container
docker build -t visual-safety-assessment .

# Run evaluation
docker run --gpus all -v $(pwd)/data:/app/data visual-safety-assessment \
    python anchor_based_safety_scoring.py --image_folder /app/data/images
```

## 🔧 Configuration

### Model Parameters

```python
# Recommended settings
BATCH_SIZE = 1
LEARNING_RATE = 5e-5
MAX_LENGTH = 512
GRADIENT_ACCUMULATION = 8
EPOCHS = 3
```

### Training Data Split

```python
# Default split
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.2
```

## 🧪 Testing

```bash
# Run comprehensive evaluation
python anchor_based_safety_scoring.py \
    --image_folder test_images \
    --max_comparisons 20 \
    --output_dir test_results

# Validate against ground truth
python evaluation_metrics.py \
    --training_data test_annotations.json \
    --vlm_results test_results/anchor_safety_evaluation_report.json
```

## 📚 References

1. **Qwen-VL**: Multi-modal Vision-Language Learning
2. **Pairwise Comparison**: Thurstone's Law of Comparative Judgment
3. **Safety Assessment**: Computer Vision for Urban Planning
4. **Score Aggregation**: Bradley-Terry Models for Preference Learning

## 🎟 Citation

If you use this work, please cite:

```bibtex
@article{visual_safety_2024,
  title={Anchor-Based Safety Assessment of Street View Images using Vision-Language Models},
  author={Your Name},
  journal={Computer Vision and Urban Safety},
  year={2024}
}
```

## 📞 Support

- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions  
- **Contact**: your.email@domain.com

## 📄 License

MIT License - see LICENSE file for details.

---

**⚠️ Note**: This project focuses exclusively on visual street view image analysis. All audio-related functionality has been removed to align with the visual-only safety assessment objective.
