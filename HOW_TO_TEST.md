# How to Test the Prompt Fixes

## Quick Validation Test

Run the validation test to verify the fixes work:

```bash
cd "C:\Users\enoch\OneDrive\Documents\Safer Place\Soundscape-to-Image"
python test_prompt_fixes.py
```

**Expected Output**: All tests should show `[PASS]`

---

## Full Model Evaluation Test

### Prerequisites
1. Activate the appropriate virtual environment:
   ```bash
   .\activate_qwen.bat
   ```

2. Ensure you have a fine-tuned model at: `models/fine_tuned_qwen2vl_trained`

### Run Evaluation

**Test on a few images first:**
```bash
python scripts/evaluation/fine_tuned_eval.py --model_path models/fine_tuned_qwen2vl_trained --image_folder data/images/image --output results/evaluation_test.xlsx
```

**Full evaluation:**
```bash
python scripts/evaluation/fine_tuned_eval.py --model_path models/fine_tuned_qwen2vl_trained --image_folder data/images/image --output results/evaluation_fixed.xlsx
```

### What to Check in Results

Open the output Excel file (`results/evaluation_fixed.xlsx`) and verify:

1. **Main Sheet - "Evaluation Results"**:
   - `Safety Score` column: Should contain numbers (1-10) or "N/A"
   - `Risk Level` column: Should contain HIGH/MEDIUM/LOW or "N/A"
   - `Score Response`: Should be clean, focused text (no rambling)
   - `Risk Response`: Should have structured risk categories
   - `Has Risk Level`: Should mostly show "Yes"

2. **Statistics Sheet**:
   - `Successful Risk Levels`: Should be high (close to total images)
   - `Failed Risk Levels`: Should be low or 0

3. **Check for Issues**:
   - Look at responses - should NOT contain:
     - "Example response..."
     - "For example..."
     - "I'm sorry but..."
     - "So far so good..."
     - "OK OK, let's get started..."
   - Responses should be concise and focused

---

## Expected Improvements

### Before (Old Output Example):
```
"Example response (if needed): Safety score for this image...
I'm sorry but we currently don't have any ratings...
So far so good! The rest should be straightforward..."
```

### After (New Output Example):
```
Safety Score: 7
Reasoning: Wide sidewalks and well-maintained infrastructure. 
Low traffic volume. Main concern is limited nighttime lighting.
```

---

## Troubleshooting

### If validation test fails:
- Check Python version (requires Python 3.8+)
- Ensure dependencies are installed: `pip install -r qwen_requirements.txt`

### If evaluation fails:
- Verify model path exists: `models/fine_tuned_qwen2vl_trained`
- Check image folder has .jpg or .png files
- Ensure enough GPU memory (or use CPU with `--device cpu`)

### If outputs still have rambling:
- Check generation parameters in `fine_tuned_eval.py` lines 236-251
- Verify `clean_response()` function is being called (line 257)
- Review stop_phrases list (lines 76-88)

---

## Quick Test Summary

✅ **Risk Extraction**: Working - extracts HIGH/MEDIUM/LOW  
✅ **Rambling Detection**: Working - removes conversational text  
✅ **Template Removal**: Working - filters instructional text  
✅ **Focused Output**: Working - uses greedy decoding + penalties


