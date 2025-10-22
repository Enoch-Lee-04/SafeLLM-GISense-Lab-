# LLM Prompt and Output Fixes

## Issues Fixed

### 1. Missing Risk Evaluation Outputs
**Problem**: Risk assessment prompt was too simplistic, didn't match training data format.

**Solution**: Updated risk prompt to match training data structure:
```python
"risks": "<|image_pad|>Assess safety risks by category. Format: 'HIGH RISK: [elements]\\nMEDIUM RISK: [elements]\\nLOW RISK: [elements]\\nOverall Risk Level: [LOW/MEDIUM/HIGH]\\nPrimary Risk Factor: [concern]'"
```

Added `extract_risk_level()` function to parse risk outputs and included risk evaluation in results.

### 2. Random Useless Output (Rambling)
**Problem**: Model generated conversational/hypothetical text like "I'm sorry but...", "So far so good!", "For example...", etc.

**Solution**:
- Changed generation from sampling to **greedy decoding** (`do_sample=False`)
- Increased **repetition_penalty** to 1.8
- Reduced **max_new_tokens** to 100 (strict limit)
- Set **length_penalty** to 0.8 (prefer shorter responses)
- Enhanced `clean_response()` to detect and stop at rambling phrases:
  - "i'm sorry but", "i'm not sure", "wouldn't it make sense"
  - "is my feeling right", "so far so good", "ok ok"
  - "let's get started", "what do you think", etc.
- Removed template text patterns: "[number]", "[reason]", "Example response:", "For example,", etc.

## Changes Made

**File**: `Soundscape-to-Image/scripts/evaluation/fine_tuned_eval.py`

1. **Updated prompts** (lines 220-228): All prompts now match training data format
2. **Added `extract_risk_level()`** (lines 100-114): Extracts HIGH/MEDIUM/LOW risk levels
3. **Improved generation parameters** (lines 236-251): Greedy decoding with strong repetition penalty
4. **Enhanced cleaning** (lines 45-110): Filters template text and stops at rambling phrases
5. **Added risk evaluation** (lines 317-321, 330-331): Risk assessment now included in results
6. **Updated stats** (lines 383-384, 393-394, 430-431): Track risk level success rate

## Usage

Run evaluation with fixed prompts:
```bash
python scripts/evaluation/fine_tuned_eval.py --model_path models/fine_tuned_qwen2vl_trained --image_folder data/images/image --output results/evaluation_fixed.xlsx
```

Results now include:
- Safety Score
- Binary Classification  
- **Risk Level** (NEW)
- **Risk Response** (NEW)
- All with cleaner, more focused outputs

