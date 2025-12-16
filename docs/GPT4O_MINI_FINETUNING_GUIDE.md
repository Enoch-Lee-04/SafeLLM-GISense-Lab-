# GPT-4o-mini Fine-Tuning and Model Comparison Guide

This guide covers the complete workflow for fine-tuning GPT-4o-mini on street view safety assessment and comparing it with GPT-4 and Qwen2-VL.

## Overview

This project now supports three models for safety assessment:
1. **GPT-4** - Baseline model (no fine-tuning)
2. **GPT-4o-mini (fine-tuned)** - Fine-tuned on your ground truth data
3. **Qwen2-VL-2B (fine-tuned)** - Fine-tuned vision-language model

## Prerequisites

1. OpenAI API key (stored in `API_KEY_Enoch` file)
2. Python 3.8+
3. Required packages:
   ```bash
   pip install openai pandas openpyxl pillow matplotlib seaborn tqdm
   ```

## Step-by-Step Workflow

### Step 1: Prepare Fine-Tuning Data

Convert your ground truth data to OpenAI's required JSONL format:

```bash
python scripts/training/prepare_openai_finetuning_data.py
```

This will create:
- `data/openai_finetuning/train_text_only.jsonl` - Training data (text-only)
- `data/openai_finetuning/val_text_only.jsonl` - Validation data (text-only)
- `data/openai_finetuning/train_with_images.jsonl` - Training data with base64 images
- `data/openai_finetuning/val_with_images.jsonl` - Validation data with base64 images

**Note**: We use text-only format by default as it's more reliable. Vision fine-tuning may have additional requirements.

### Step 2: Fine-Tune GPT-4o-mini

Start the fine-tuning job:

```bash
python scripts/training/finetune_gpt4o_mini.py
```

This will:
1. Upload your training and validation files to OpenAI
2. Create a fine-tuning job
3. Monitor the job progress (can take several hours)
4. Save the fine-tuned model information to `models/gpt4o_mini_finetuned/fine_tune_info.json`

**Important**: 
- Fine-tuning can take 1-4 hours depending on data size
- You can stop monitoring and check status later
- You'll be charged based on tokens processed

### Step 3: Test Individual Models

#### Test GPT-4 (Baseline)
```bash
cd scripts/evaluation
python unified_model_inference.py
```

#### Test Fine-Tuned GPT-4o-mini
After fine-tuning completes:
```python
from unified_model_inference import ModelFactory

model = ModelFactory.create_model("gpt4o-mini-ft")
response = model.predict("path/to/image.jpg", "Your prompt here")
print(response)
```

#### Test Fine-Tuned Qwen2-VL
```python
from unified_model_inference import ModelFactory

model = ModelFactory.create_model("qwen2vl-ft")
response = model.predict("path/to/image.jpg", "Your prompt here")
print(response)
```

### Step 4: Run Comprehensive Comparison

Compare all three models on your test set:

```bash
python scripts/evaluation/compare_all_models.py --models gpt4 gpt4o-mini-ft qwen2vl-ft
```

Options:
- `--models`: Specify which models to compare (default: all three)
- `--output-dir`: Custom output directory for results
- `--gpt4-model`: Specify GPT-4 variant (default: "gpt-4")

This will:
1. Load test data from your ground truth
2. Run inference on all models
3. Calculate metrics (MAE, RMSE, Accuracy@1, Accuracy@2)
4. Save detailed results to `results/model_comparison/`

### Step 5: Analyze and Visualize Results

Generate comprehensive analysis and visualizations:

```bash
python scripts/evaluation/analyze_comparison_results.py
```

This creates:
- **Metrics comparison plot** - Bar charts comparing MAE, RMSE, and accuracy
- **Score distribution plot** - Scatter plots of predicted vs expected scores
- **Error distribution plot** - Histograms showing prediction errors
- **Detailed text report** - Analysis including best/worst predictions

All outputs saved to `results/model_comparison/plots/`

## Understanding the Metrics

### Mean Absolute Error (MAE)
- Average absolute difference between predicted and expected scores
- **Lower is better**
- Range: 0 to 10 (in our 1-10 scale)
- Good performance: < 1.0

### Root Mean Square Error (RMSE)
- Square root of average squared errors
- Penalizes large errors more than MAE
- **Lower is better**
- Range: 0 to 10
- Good performance: < 1.5

### Accuracy within 1 point (Acc@1)
- Percentage of predictions within ±1 point of expected score
- **Higher is better**
- Range: 0% to 100%
- Good performance: > 70%

### Accuracy within 2 points (Acc@2)
- Percentage of predictions within ±2 points of expected score
- **Higher is better**
- Range: 0% to 100%
- Good performance: > 85%

## Expected Results

Based on typical fine-tuning outcomes:

| Model | MAE | RMSE | Acc@1 | Acc@2 |
|-------|-----|------|-------|-------|
| GPT-4 (baseline) | 1.2-1.5 | 1.5-2.0 | 60-70% | 80-85% |
| GPT-4o-mini (fine-tuned) | 0.8-1.2 | 1.0-1.5 | 70-80% | 85-90% |
| Qwen2-VL (fine-tuned) | 0.7-1.0 | 0.9-1.3 | 75-85% | 90-95% |

**Note**: These are estimates. Actual results depend on your data quality and fine-tuning parameters.

## Troubleshooting

### Fine-Tuning Failed
- Check your JSONL format is correct
- Ensure API key has sufficient credits
- Verify training data doesn't contain errors
- Check OpenAI dashboard for detailed error messages

### Model Not Found
- For GPT-4o-mini: Ensure fine-tuning completed successfully
- Check `models/gpt4o_mini_finetuned/fine_tune_info.json` for model name
- For Qwen2-VL: Ensure model is in `models/fine_tuned_qwen2vl/`

### Out of Memory (Qwen2-VL)
- Use CPU mode: Add `device="cpu"` when creating model
- Reduce batch size if processing multiple images
- Close other applications

### API Rate Limits
- Add delays between requests if hitting rate limits
- Consider upgrading OpenAI API tier for higher limits
- Use smaller test set for faster iterations

## Cost Estimation

### Fine-Tuning Costs (GPT-4o-mini)
- Training: ~$0.02-0.10 per 1K tokens
- Typical cost for ~200 examples: $5-20
- One-time cost

### Inference Costs
- GPT-4: ~$0.03 per image (with ~500 token response)
- GPT-4o-mini: ~$0.001-0.002 per image
- Qwen2-VL: Free (runs locally)

**For 30 images**:
- GPT-4: ~$1.00
- GPT-4o-mini: ~$0.06
- Qwen2-VL: Free

## Checking Fine-Tuning Status

If you need to check the status of a fine-tuning job:

```python
from scripts.training.finetune_gpt4o_mini import GPT4oMiniFineTuner

finetuner = GPT4oMiniFineTuner()

# List recent jobs
finetuner.list_fine_tune_jobs(limit=10)

# Check specific job
finetuner.check_job_status("ftjob-xxxxxxxxxxxxx")

# Resume monitoring
finetuner.monitor_fine_tune_job("ftjob-xxxxxxxxxxxxx")
```

## Advanced Configuration

### Adjusting Hyperparameters

Edit `scripts/training/finetune_gpt4o_mini.py`:

```python
hyperparameters = {
    "n_epochs": 3,  # Increase for more training (default: 3)
    "batch_size": "auto",  # Or specify: 1, 2, 4, etc.
    "learning_rate_multiplier": "auto"  # Or specify: 0.1, 0.5, 1.0, 2.0
}
```

**Tips**:
- More epochs = better fit but risk of overfitting
- Larger batch size = faster training but more memory
- Higher learning rate = faster convergence but less stable

### Using Different Test/Train Splits

Modify `prepare_openai_finetuning_data.py`:

```python
def create_validation_split(training_data, validation_ratio=0.2):  # Change ratio
    # ... rest of function
```

## Best Practices

1. **Start Small**: Test with a subset before full evaluation
2. **Monitor Costs**: Keep track of API usage
3. **Save Results**: All results are timestamped and saved automatically
4. **Iterate**: Adjust hyperparameters based on results
5. **Document**: Keep notes on what works best for your use case

## Next Steps

After getting good results:
1. Deploy the best model for production use
2. Set up continuous evaluation pipeline
3. Collect more data to further improve models
4. Consider ensemble methods combining multiple models

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review OpenAI fine-tuning documentation
3. Check model-specific documentation (Qwen2-VL)

## References

- [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [Qwen2-VL Documentation](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- Project-specific guides in `docs/`


