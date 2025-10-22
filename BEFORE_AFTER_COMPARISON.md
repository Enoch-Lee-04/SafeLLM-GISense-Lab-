# Before/After Comparison

## Issue 1: Missing Risk Evaluation Outputs

### BEFORE
```python
"risks": "<|image_pad|>Assess risk level: HIGH, MEDIUM, or LOW. Answer format: 'Risk Level: [level]'"
```
- Too simplistic
- Doesn't match training data format
- Missing risk categorization
- No risk assessment details

### AFTER
```python
"risks": "<|image_pad|>Assess safety risks by category. Format: 'HIGH RISK: [elements]\\nMEDIUM RISK: [elements]\\nLOW RISK: [elements]\\nOverall Risk Level: [LOW/MEDIUM/HIGH]\\nPrimary Risk Factor: [concern]'"
```
- Matches training data structure
- Includes all risk categories
- Requests overall risk level
- Asks for primary risk factor
- Added `extract_risk_level()` function to parse outputs

---

## Issue 2: Random Useless Output

### BEFORE - Example Rambling Output
```
"Example response (if needed): Safety score for this image,
considering all factors combined into an overall assessment that I
can rate on my scale above:

Safe

Moderate Risk

Not Safe - Please do not use.

I'm sorry but we currently don't have any ratings or reviews
available in our system to allow us providing safe scores here.

So far so good! The rest should be straightforward now with no
need to refer back again if there are issues like rating missing during
review process andwhatnots...

OK OK, let's get started!

The first thing you'll notice about this picture?"
```

### AFTER - Clean Focused Output
```
Safety Score: 7
Reasoning: The sidewalk is wide and well maintained, traffic seems low. 
Main drawback is absence of visible streetlights and lack of buffer 
between pedestrians and vehicles.
```

### Changes to Achieve This

#### Generation Parameters
**BEFORE**:
```python
do_sample=True
temperature=0.6
top_p=0.85
top_k=40
max_new_tokens=120
repetition_penalty=1.5
```

**AFTER**:
```python
do_sample=False          # Greedy decoding - no sampling randomness
temperature=1.0          # Not used with greedy
num_beams=1             # Efficient single beam
max_new_tokens=100      # Stricter limit
repetition_penalty=1.8  # Stronger penalty
no_repeat_ngram_size=5  # Prevent 5-gram repetition
length_penalty=0.8      # Prefer shorter responses
early_stopping=True     # Stop at EOS token
```

#### Response Cleaning
**ADDED**: Stop phrases detection
```python
stop_phrases = [
    "i'm sorry but",
    "i'm not sure",
    "wouldn't it make sense",
    "is my feeling right",
    "based on",
    "so far so good",
    "ok ok",
    "let's get started",
    "the first thing you'll notice",
    "what do you think",
    "how would you feel"
]
```

**ADDED**: Template text removal
```python
template_patterns = [
    r'\[number\]',
    r'\[reason\]',
    r'\[level\]',
    r'\[concern\]',
    r'\[elements\]',
    r'Example response.*?:',
    r'For example.*?[:,]',
    r'Answer format.*?:',
    r'Response format.*?:',
    r'Format:.*?\n',
]
```

---

## Results Tracking

### NEW Columns Added to Excel Output:
- `Risk Level`: Extracted risk level (HIGH/MEDIUM/LOW)
- `Risk Response`: Full risk assessment response
- `Has Risk Level`: Success indicator for risk extraction

### NEW Statistics:
- Successful Risk Levels count
- Failed Risk Levels count

---

## Summary

✅ **Fixed**: Missing risk evaluation outputs  
✅ **Fixed**: Random rambling responses  
✅ **Improved**: All prompts now match training data format  
✅ **Improved**: Response cleaning removes template text and stops rambling  
✅ **Improved**: Generation parameters favor focused, concise outputs

