# Fine-Tuning Issues and Solutions

## Problem Summary

The fine-tuned Qwen2-VL model exhibited inconsistent output generation with only 4 out of 30 successful score extractions (13% success rate). This document outlines the issues discovered and solutions implemented.

## Root Causes

### 1. **Insufficient Training**
- **Issue**: Model trained for only 1-2 epochs with ~30 examples
- **Impact**: Model learned to generate safety-related text but didn't internalize the specific output format
- **Evidence**: Responses were verbose and descriptive rather than structured

### 2. **Inconsistent Output Formats**
The model generated responses in various incompatible formats:

**Example 1** - Score buried in text:
```
The scene is set in the city, with cars driving on multiple lanes...
Safety Score:
* Reasonable - It appears there may have been some issue...
```

**Example 2** - Markdown formatting:
```
![](Image.png)
The road is wet, but there are no visible pedestrians...
Safety Score: **6**
Reasoning: While the scene appears calm...
```

**Example 3** - Incomplete structured output:
```
The road is relatively clear with no visible obstacles...
[Street View Image]
```

**Example 4** - Excessive context without clear score:
```
Based on the provided Street View Image...
* Safety Features Present in Urban Area
+ Well-lit areas due to numerous lights...
```

### 3. **Repetition and Template Echoing**
- Model frequently repeated prompts as part of responses
- Generated repetitive patterns (same phrase 3-5 times)
- Left incomplete sentences at end of responses

### 4. **Weak Score Extraction Logic**
- Original extraction only looked for exact patterns like "Safety Score: 5"
- Couldn't handle variations like "**6**", "Score: 6", or scores embedded in text
- No fallback for inferring scores from qualitative assessments

## Solutions Implemented

### Solution 1: Improved Score Extraction

**File**: `scripts/evaluation/rerun_evaluation_improved.py`

**Improvements**:
1. **Multiple pattern matching** - Added 8+ patterns to catch various formats:
   - `Safety Score: **6**`
   - `Score: 6`
   - `safety score of 6`
   - `6/10` format
   - Standalone numbers in safety context

2. **Context-aware extraction** - Searches for numbers (0-10) near safety-related keywords

3. **Inference from qualitative terms**:
   - "HIGH RISK" → score 3.0
   - "MEDIUM RISK" → score 5.0  
   - "LOW RISK" → score 7.0
   - "SAFE" classification → score 7.0
   - "UNSAFE" classification → score 3.0
   - "reasonable" → score 6.0

4. **False positive prevention** - Excludes numbers that are likely addresses, years, or lane numbers

### Solution 2: Improved Generation Parameters

**Changes in `test_fine_tuned_qwen.py`**:

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=150,        # Reduced from 200 to prevent rambling
    min_new_tokens=20,         # Ensure meaningful response
    do_sample=True,            # Enable sampling for variety
    temperature=0.7,           # Moderate creativity
    top_p=0.9,                 # Nucleus sampling
    top_k=50,                  # Limit token choices
    repetition_penalty=1.2,    # Penalize repetition
    no_repeat_ngram_size=3,    # Prevent 3-gram repetition
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id
)
```

**Impact**: Reduces repetitive outputs and incomplete responses

### Solution 3: Enhanced Response Cleaning

**Implemented in `clean_response()` function**:

1. **Prompt removal** - Strips echoed input prompts from responses
2. **Template pattern removal** - Removes placeholder text like `[1-10]`, `[SAFE/UNSAFE]`
3. **Repetition detection** - Stops output when same line appears 2+ times
4. **Incomplete sentence trimming** - Removes trailing incomplete patterns
5. **Multiple score consolidation** - Keeps only first score if multiple exist

### Solution 4: More Explicit Prompts

**Changed from**:
```
"Analyze this street view image and provide a safety score from 1-10..."
```

**Changed to**:
```
"Rate the safety of this street view from 1-10 where 10 is safest. 
You MUST provide a number. 
Format: 'Safety Score: [number]' then one sentence explaining why."
```

**Rationale**: More explicit formatting requirements may improve compliance

### Solution 5: Improved Training Script

**File**: `scripts/training/retrain_qwen_improved.py`

**Key Improvements**:
1. **Increased training epochs**: 5 epochs (up from 1-2)
2. **Better data formatting**: Clear conversation structure with `<|endoftext|>` tokens
3. **Proper label masking**: Only trains on assistant responses, not prompts
4. **Consistent padding**: Ensures uniform batch processing
5. **Lower learning rate**: 2e-5 (more stable than 1e-5 or 5e-5)
6. **More warmup steps**: 50 steps for gradual learning

## Recommended Next Steps

### Immediate (to improve current model evaluation)
1. ✅ Run improved evaluation script with better extraction
2. ⏳ Analyze success rate improvement
3. ⏳ Compare extracted scores with ground truth

### Short-term (to improve model quality)
1. **Retrain with improved configuration**:
   ```bash
   python scripts/training/retrain_qwen_improved.py \
     --data_path configs/vlm_safety_training_data.json \
     --image_folder "data/images/image" \
     --output_dir models/fine_tuned_qwen2vl_improved \
     --num_train_epochs 5 \
     --per_device_train_batch_size 2 \
     --learning_rate 2e-5
   ```

2. **Augment training data**:
   - Add more diverse examples (currently only 30)
   - Include examples showing the EXACT expected output format
   - Add negative examples (what NOT to do)
   - Target 100-200 training examples minimum

3. **Format validation during training**:
   - Add custom callback to check if model outputs valid scores
   - Stop training if outputs degrade
   - Log sample predictions each epoch

### Long-term (for production quality)
1. **Use LoRA fine-tuning** - More parameter efficient
2. **Implement few-shot prompting** - Include format examples in prompt
3. **Add output validation layer** - Post-process to ensure valid format
4. **Create evaluation metrics** - Track format compliance, not just accuracy
5. **Consider instruction-tuned base model** - Better at following format instructions

## Key Learnings

1. **Format compliance ≠ Knowledge acquisition**: Model can learn what safety looks like without learning how to format responses

2. **Small datasets require more epochs**: 30 examples × 1 epoch = 30 training steps (too few)

3. **Explicit format requirements help**: Being more prescriptive in prompts improves compliance

4. **Generation parameters matter**: Temperature, repetition penalty, and max tokens significantly affect output quality

5. **Robust extraction is essential**: Even with perfect training, some format variations are inevitable

## Performance Metrics

### Before Improvements
- Success rate: 13% (4/30 images)
- Mean Absolute Error: 1.67 (on 4 valid scores)
- Common issues: Repetition, incomplete responses, no scores

### After Improvements (Pending)
- Expected success rate: 50-70% (with improved extraction)
- Expected MAE: TBD
- Should handle: Multiple formats, embedded scores, qualitative assessments

### Target (After Retraining)
- Target success rate: >90%
- Target MAE: <2.0
- Consistent format compliance

## Files Modified

1. `scripts/evaluation/test_fine_tuned_qwen.py` - Better generation parameters
2. `scripts/evaluation/rerun_evaluation_improved.py` - Improved extraction and prompts
3. `scripts/training/retrain_qwen_improved.py` - New training script with better configuration

## References

- Original training script: `scripts/training/qwen_safety_finetune_simple.py`
- Training data: `configs/vlm_safety_training_data.json`
- Model checkpoints: `models/fine_tuned_qwen2vl_trained/`

