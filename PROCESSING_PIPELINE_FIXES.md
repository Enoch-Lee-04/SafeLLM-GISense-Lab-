# Fixed Qwen-VL Processing Pipeline

This document explains the fixes made to resolve the processing pipeline issues encountered during the initial training run.

## Problem Summary

The initial training run completed successfully (27 epochs), but the evaluation revealed critical technical issues:

1. **Missing Method Error**: `'Qwen2VLProcessor' object has no attribute 'process_multimodal_inputs'`
2. **Token Mismatch Error**: `Image features and image tokens do not match: tokens: 0, features 392/196`
3. **Complete Evaluation Failure**: All images received safety scores of 0.0

## Root Cause Analysis

The issues were caused by **incorrect usage of the Qwen2VLProcessor**:

1. **Wrong Method Call**: The evaluation scripts were trying to use `process_multimodal_inputs()` method which doesn't exist in Qwen2VLProcessor
2. **Incorrect Input Format**: The scripts were using `conversation` parameter instead of the correct `text` and `images` parameters
3. **Missing Image Tokens**: The processor wasn't receiving properly formatted text with image token placeholders

## Solution Implementation

### 1. Correct Processing Method

**Before (Incorrect):**
```python
# This doesn't work - method doesn't exist
inputs = self.processor.process_multimodal_inputs(
    conversation=conversation,
    return_tensors="pt"
)
```

**After (Correct):**
```python
# Use the correct __call__ method with proper parameters
text_with_image = f"<|image_pad|>{prompt}"
inputs = self.processor(
    text=text_with_image,
    images=image,
    return_tensors="pt",
    padding=True,
    truncation=True
)
```

### 2. Proper Image Token Handling

The Qwen2VLProcessor automatically handles image tokens when you:
- Include `<|image_pad|>` placeholders in the text
- Pass images as a separate parameter
- The processor automatically replaces placeholders with the correct number of tokens

### 3. Correct Response Decoding

**Before (Incorrect):**
```python
response = self.processor.decode(outputs[0], skip_special_tokens=True)
```

**After (Correct):**
```python
# Extract only the generated response (exclude input)
input_length = len(inputs['input_ids'][0])
response = self.processor.tokenizer.decode(
    outputs[0][input_length:], 
    skip_special_tokens=True
)
```

## Fixed Files

### 1. `qwen_safety_inference_fixed.py`
- Fixed single image safety assessment
- Fixed pairwise image comparison
- Correct processing pipeline implementation
- Proper error handling

### 2. `scripts/evaluation/vlm_baseline_evaluation_fixed.py`
- Fixed baseline evaluation with correct processing
- Proper pairwise comparison implementation
- Correct response parsing

### 3. `scripts/evaluation/anchor_based_safety_scoring_fixed.py`
- Fixed anchor-based scoring system
- Correct multimodal processing
- Proper comparison result parsing

### 4. `test_fixed_pipeline.py`
- Comprehensive test suite
- Validates processing pipeline components
- Tests single image and pairwise comparison functionality

## Usage Instructions

### Test the Fixed Pipeline
```bash
python test_fixed_pipeline.py
```

### Run Single Image Assessment
```bash
python qwen_safety_inference_fixed.py \
    --model_path "path/to/your/model" \
    --image_path "path/to/image.jpg" \
    --prompt_type "safety_score"
```

### Run Baseline Evaluation
```bash
python scripts/evaluation/vlm_baseline_evaluation_fixed.py \
    --image_folder "data/samples/Sample SVI" \
    --output_dir "./fixed_evaluation_results" \
    --model_name "Qwen/Qwen2-VL-2B-Instruct"
```

### Run Anchor-Based Scoring
```bash
python scripts/evaluation/anchor_based_safety_scoring_fixed.py \
    --image_folder "data/samples/Sample SVI" \
    --output_dir "./fixed_anchor_results" \
    --model_name "Qwen/Qwen2-VL-2B-Instruct"
```

## Key Technical Changes

### 1. Input Processing
- **Single Image**: `<|image_pad|>{prompt}` + `images=image`
- **Multiple Images**: `<|image_pad|><|image_pad|>{prompt}` + `images=[img1, img2]`

### 2. Response Generation
- Proper device handling for CUDA/CPU
- Correct token ID handling
- Proper response length management

### 3. Error Handling
- Comprehensive exception handling
- Detailed error messages
- Graceful degradation

## Expected Results

With the fixed pipeline, you should now see:

1. **Successful Image Processing**: No more token mismatch errors
2. **Valid Safety Scores**: Scores ranging from 0-10 instead of all 0.0
3. **Proper Comparisons**: Successful pairwise image comparisons
4. **Meaningful Responses**: Actual safety assessments instead of error messages

## Verification Steps

1. **Run Test Suite**: `python test_fixed_pipeline.py`
2. **Check Processing**: Verify no token mismatch errors
3. **Validate Scores**: Ensure safety scores are non-zero
4. **Test Comparisons**: Verify pairwise comparisons work
5. **Review Output**: Check that responses contain actual assessments

## Next Steps

1. **Re-run Evaluation**: Use the fixed scripts to properly evaluate your trained model
2. **Compare Results**: Compare fixed results with the synthetic baseline
3. **Analyze Performance**: Assess actual model performance vs ground truth
4. **Iterate Training**: Use evaluation results to improve future training runs

The training itself was successful - the issue was purely in the inference/evaluation pipeline. With these fixes, you can now properly assess your model's performance.
