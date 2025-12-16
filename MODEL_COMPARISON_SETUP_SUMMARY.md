# Model Comparison Setup - Complete Summary

## What Was Created

This document summarizes everything that was set up for comparing GPT-4, GPT-4o-mini (fine-tuned), and Qwen2-VL (fine-tuned) models on your street view safety assessment task.

---

## 📁 New Files Created

### 1. Training Scripts

#### `scripts/training/prepare_openai_finetuning_data.py`
**Purpose**: Convert your ground truth data to OpenAI's fine-tuning format (JSONL)

**Features**:
- Converts `vlm_safety_training_data.json` to JSONL format
- Creates both image-based and text-only versions
- Automatically splits data into train/validation sets (90/10)
- Handles image encoding (base64) for vision models
- Creates 4 output files in `data/openai_finetuning/`

**Usage**:
```bash
python scripts/training/prepare_openai_finetuning_data.py
```

**Output**:
- `train_with_images.jsonl` - Training data with base64 images
- `val_with_images.jsonl` - Validation data with base64 images
- `train_text_only.jsonl` - Training data (text-only, more reliable)
- `val_text_only.jsonl` - Validation data (text-only)

---

#### `scripts/training/finetune_gpt4o_mini.py`
**Purpose**: Fine-tune GPT-4o-mini using OpenAI's API

**Features**:
- Uploads training/validation files to OpenAI
- Creates and monitors fine-tuning jobs
- Saves fine-tuned model info for later use
- Supports job status checking and resuming
- Lists recent fine-tuning jobs

**Usage**:
```bash
# Start fine-tuning
python scripts/training/finetune_gpt4o_mini.py

# Check status in Python
from scripts.training.finetune_gpt4o_mini import GPT4oMiniFineTuner
finetuner = GPT4oMiniFineTuner()
finetuner.check_job_status("ftjob-xxxxx")
```

**Output**:
- `models/gpt4o_mini_finetuned/fine_tune_info.json` - Model metadata

**Time**: 1-4 hours depending on data size
**Cost**: $5-20 for ~200 training examples

---

### 2. Evaluation Scripts

#### `scripts/evaluation/unified_model_inference.py`
**Purpose**: Unified interface for all three models

**Features**:
- Common API for GPT-4, GPT-4o-mini, and Qwen2-VL
- `ModelFactory` for easy model creation
- Handles image encoding and processing for each model
- Base classes for extensibility

**Classes**:
- `BaseModelInference` - Abstract base class
- `GPT4Inference` - GPT-4 baseline inference
- `GPT4oMiniFineTunedInference` - Fine-tuned GPT-4o-mini
- `Qwen2VLFineTunedInference` - Fine-tuned Qwen2-VL
- `ModelFactory` - Creates model instances

**Usage**:
```python
from unified_model_inference import ModelFactory

# Create any model
model = ModelFactory.create_model("gpt4")  # or "gpt4o-mini-ft", "qwen2vl-ft"

# Get prediction
response = model.predict("path/to/image.jpg", "Your prompt here")
```

---

#### `scripts/evaluation/compare_all_models.py`
**Purpose**: Comprehensive comparison of all models

**Features**:
- Evaluates all models on same test set
- Extracts safety scores from responses (multiple formats)
- Calculates metrics: MAE, RMSE, Accuracy@1, Accuracy@2
- Saves detailed results (JSON and Excel)
- Generates comparison summary

**Classes**:
- `SafetyScoreExtractor` - Extracts scores from responses
- `ModelComparator` - Orchestrates comparison

**Usage**:
```bash
# Compare all models
python scripts/evaluation/compare_all_models.py

# Compare specific models
python scripts/evaluation/compare_all_models.py --models gpt4 qwen2vl-ft

# Custom output directory
python scripts/evaluation/compare_all_models.py --output-dir custom/path
```

**Output**:
- `results/model_comparison/[model]_results_[timestamp].json` - Detailed results
- `results/model_comparison/[model]_results_[timestamp].xlsx` - Excel format
- `results/model_comparison/model_comparison_[timestamp].json` - Summary
- `results/model_comparison/metrics_comparison_[timestamp].xlsx` - Metrics table

---

#### `scripts/evaluation/analyze_comparison_results.py`
**Purpose**: Visualize and analyze comparison results

**Features**:
- Creates publication-quality plots
- Generates detailed text report
- Analyzes best/worst predictions
- Shows error distributions

**Visualizations Created**:
1. **Metrics comparison** - Bar charts for MAE, RMSE, Acc@1, Acc@2
2. **Score distribution** - Scatter plots (predicted vs expected)
3. **Error distribution** - Histograms of prediction errors

**Usage**:
```bash
python scripts/evaluation/analyze_comparison_results.py
```

**Output**:
- `results/model_comparison/plots/metrics_comparison_[timestamp].png`
- `results/model_comparison/plots/score_distribution_[timestamp].png`
- `results/model_comparison/plots/error_distribution_[timestamp].png`
- `results/model_comparison/detailed_report_[timestamp].txt`

---

### 3. Pipeline Orchestration

#### `run_model_comparison_pipeline.py`
**Purpose**: Master script to run entire pipeline

**Features**:
- Interactive or automated mode
- Step-by-step execution with error handling
- Skip completed steps
- Customizable model selection
- Progress tracking

**Usage**:
```bash
# Interactive mode (recommended for first time)
python run_model_comparison_pipeline.py

# Automatic mode (no prompts)
python run_model_comparison_pipeline.py --auto

# Skip already-completed steps
python run_model_comparison_pipeline.py --skip-data-prep --skip-finetuning

# Compare only specific models
python run_model_comparison_pipeline.py --models gpt4 qwen2vl-ft
```

**Steps Executed**:
1. Data preparation
2. GPT-4o-mini fine-tuning (optional)
3. Model comparison
4. Results analysis

---

### 4. Documentation

#### `docs/GPT4O_MINI_FINETUNING_GUIDE.md`
**Comprehensive guide covering**:
- Complete workflow explanation
- Step-by-step instructions
- Metrics interpretation
- Troubleshooting common issues
- Cost estimation
- Best practices
- Advanced configuration

---

#### `QUICKSTART_MODEL_COMPARISON.md`
**Quick reference guide with**:
- 3 ways to run the pipeline
- Installation instructions
- Expected results
- Quick commands
- Troubleshooting tips
- Success metrics

---

## 🎯 Complete Workflow

```
1. Prepare Data
   ↓
   [Ground Truth JSON] → prepare_openai_finetuning_data.py → [JSONL Files]

2. Fine-Tune
   ↓
   [JSONL Files] → finetune_gpt4o_mini.py → [Fine-tuned Model on OpenAI]

3. Compare
   ↓
   [Test Images] → compare_all_models.py → [Results JSON/Excel]
   │
   ├─ GPT-4 (baseline)
   ├─ GPT-4o-mini (fine-tuned)
   └─ Qwen2-VL (fine-tuned)

4. Analyze
   ↓
   [Results] → analyze_comparison_results.py → [Plots + Reports]
```

---

## 📊 Metrics Explained

### Mean Absolute Error (MAE)
- **Definition**: Average absolute difference between predicted and expected scores
- **Formula**: `(1/n) * Σ|predicted - expected|`
- **Range**: 0 to 10 (for 1-10 scale)
- **Interpretation**: Lower is better
- **Good score**: < 1.0 (excellent), < 1.5 (good)

### Root Mean Square Error (RMSE)
- **Definition**: Square root of average squared errors
- **Formula**: `√((1/n) * Σ(predicted - expected)²)`
- **Range**: 0 to 10
- **Interpretation**: Lower is better, penalizes large errors more
- **Good score**: < 1.3 (excellent), < 2.0 (good)

### Accuracy within 1 Point (Acc@1)
- **Definition**: % of predictions within ±1 point of expected
- **Range**: 0% to 100%
- **Interpretation**: Higher is better
- **Good score**: > 75% (excellent), > 60% (good)

### Accuracy within 2 Points (Acc@2)
- **Definition**: % of predictions within ±2 points of expected
- **Range**: 0% to 100%
- **Interpretation**: Higher is better
- **Good score**: > 90% (excellent), > 80% (good)

---

## 💰 Cost Analysis

### One-Time Costs
| Item | Cost | Frequency |
|------|------|-----------|
| GPT-4o-mini fine-tuning (~200 examples) | $5-20 | One time per model |
| Data preparation | Free | - |
| Qwen2-VL training | Free | Already done locally |

### Per-Evaluation Costs (30 test images)
| Model | Cost per Image | Total for 30 Images | Speed |
|-------|----------------|---------------------|-------|
| GPT-4 | ~$0.03 | ~$1.00 | Fast (API) |
| GPT-4o-mini (fine-tuned) | ~$0.002 | ~$0.06 | Fast (API) |
| Qwen2-VL (fine-tuned) | Free | Free | Slower (local) |

**Total cost for complete comparison**: ~$6-22

---

## 🚀 Quick Start Commands

```bash
# Install dependencies
pip install openai pandas openpyxl pillow matplotlib seaborn tqdm transformers torch peft

# Option 1: Run everything (easiest)
python run_model_comparison_pipeline.py --auto

# Option 2: Step by step
python scripts/training/prepare_openai_finetuning_data.py
python scripts/training/finetune_gpt4o_mini.py
python scripts/evaluation/compare_all_models.py
python scripts/evaluation/analyze_comparison_results.py

# Option 3: Skip fine-tuning (if already done)
python run_model_comparison_pipeline.py --skip-finetuning
```

---

## 📂 Output Structure

```
project/
├── data/
│   └── openai_finetuning/
│       ├── train_text_only.jsonl
│       ├── val_text_only.jsonl
│       ├── train_with_images.jsonl
│       └── val_with_images.jsonl
│
├── models/
│   └── gpt4o_mini_finetuned/
│       └── fine_tune_info.json
│
└── results/
    └── model_comparison/
        ├── gpt4_results_[timestamp].json
        ├── gpt4_results_[timestamp].xlsx
        ├── gpt4o-mini-ft_results_[timestamp].json
        ├── gpt4o-mini-ft_results_[timestamp].xlsx
        ├── qwen2vl-ft_results_[timestamp].json
        ├── qwen2vl-ft_results_[timestamp].xlsx
        ├── model_comparison_[timestamp].json
        ├── metrics_comparison_[timestamp].xlsx
        ├── detailed_report_[timestamp].txt
        └── plots/
            ├── metrics_comparison_[timestamp].png
            ├── score_distribution_[timestamp].png
            └── error_distribution_[timestamp].png
```

---

## 🔧 Customization Options

### Adjust Fine-Tuning Hyperparameters
Edit `scripts/training/finetune_gpt4o_mini.py`:
```python
hyperparameters = {
    "n_epochs": 5,  # More epochs = more training
    "batch_size": 4,  # Larger = faster but more memory
    "learning_rate_multiplier": 1.0  # Adjust learning rate
}
```

### Change Train/Val Split
Edit `scripts/training/prepare_openai_finetuning_data.py`:
```python
train_data, val_data = create_validation_split(training_data, validation_ratio=0.2)
```

### Use Different Models
```bash
# Compare only specific models
python scripts/evaluation/compare_all_models.py --models gpt4 qwen2vl-ft

# Use GPT-4-turbo instead of GPT-4
python scripts/evaluation/compare_all_models.py --gpt4-model gpt-4-turbo
```

### Custom Test Set
Edit `scripts/evaluation/compare_all_models.py`:
```python
def load_test_data(self):
    # Modify this to use your custom test set
    test_data = load_custom_test_set()
    return test_data
```

---

## ✅ Verification Checklist

Before running:
- [ ] Python 3.8+ installed
- [ ] All packages installed (`pip install ...`)
- [ ] OpenAI API key in `API_KEY_Enoch`
- [ ] Ground truth data exists at `configs/vlm_safety_training_data.json`
- [ ] Sufficient disk space (~500MB for outputs)
- [ ] OpenAI account has sufficient credits ($20+ recommended)

After data preparation:
- [ ] JSONL files created in `data/openai_finetuning/`
- [ ] Files contain correct number of examples
- [ ] JSON format is valid

After fine-tuning:
- [ ] Job completed successfully
- [ ] Model name saved in `models/gpt4o_mini_finetuned/fine_tune_info.json`
- [ ] Can access model via OpenAI API

After comparison:
- [ ] Results files created for each model
- [ ] Metrics look reasonable (MAE < 3.0)
- [ ] Excel files can be opened

After analysis:
- [ ] Plots generated successfully
- [ ] Detailed report created
- [ ] Metrics show improvement over baseline

---

## 🎯 Expected Timeline

| Task | Duration | Can Parallelize? |
|------|----------|------------------|
| Data preparation | 2-5 minutes | No |
| GPT-4o-mini fine-tuning | 1-4 hours | No (runs on OpenAI) |
| Model comparison (30 images) | 10-30 minutes | Partially |
| Results analysis | 1-2 minutes | No |

**Total**: ~2-5 hours (mostly waiting for fine-tuning)

---

## 📈 Success Criteria

Your setup is working well if:
- ✅ All scripts run without errors
- ✅ MAE < 1.5 for at least one model
- ✅ Accuracy@1 > 60% for at least one model
- ✅ Fine-tuned models outperform baseline
- ✅ Visualizations show clear trends
- ✅ Cost per evaluation is acceptable

---

## 🆘 Common Issues

| Issue | Solution |
|-------|----------|
| API key not found | Create `API_KEY_Enoch` file with your key |
| Module not found | Run `pip install openai transformers torch ...` |
| CUDA out of memory | Use `device="cpu"` for Qwen2-VL |
| Fine-tuning failed | Check OpenAI dashboard for details |
| Model not found | Ensure fine-tuning completed |
| Poor results | Check data quality, try more epochs |

See `QUICKSTART_MODEL_COMPARISON.md` for detailed troubleshooting.

---

## 🎓 Learning Resources

- **OpenAI Fine-Tuning**: https://platform.openai.com/docs/guides/fine-tuning
- **Qwen2-VL**: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
- **PEFT/LoRA**: https://huggingface.co/docs/peft

---

## 📝 Notes

- Fine-tuning is **one-time cost** - model persists on OpenAI
- Qwen2-VL runs **locally** - no ongoing API costs
- Results are **timestamped** - easy to track experiments
- All scripts are **resumable** - can stop and continue
- **GPU recommended** for Qwen2-VL but not required

---

**Ready to start?** Run:
```bash
python run_model_comparison_pipeline.py
```

**Questions?** See:
- `QUICKSTART_MODEL_COMPARISON.md` - Quick reference
- `docs/GPT4O_MINI_FINETUNING_GUIDE.md` - Detailed guide
- `docs/QWEN_FINETUNING_GUIDE.md` - Qwen-specific info


