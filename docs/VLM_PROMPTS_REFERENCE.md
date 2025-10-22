# VLM Safety Analysis - Prompt Types Reference

## 🎯 **5 Core Prompt Types for VLM Fine-tuning**

### 1. **Safety Score Prompt** (1-10 Scale)
**Purpose**: Overall safety assessment with numerical scoring
**Use Case**: Regression tasks, quantitative analysis
**Output Format**: 
```
Safety Score: [1-10]
Reasoning: [Brief explanation]
```

### 2. **Binary Classification Prompt** (SAFE/UNSAFE)
**Purpose**: Simple safe/unsafe classification
**Use Case**: Binary classification tasks, quick screening
**Output Format**:
```
Classification: [SAFE/UNSAFE]
Confidence: [HIGH/MEDIUM/LOW]
Reason: [Brief explanation]
```

### 3. **Detailed Analysis Prompt** (Multi-dimensional)
**Purpose**: Comprehensive safety analysis across multiple dimensions
**Use Case**: Detailed assessment, multi-task learning
**Output Format**:
```
1. Pedestrian Safety: [1-10]
2. Traffic Safety: [1-10]
3. Lighting Safety: [1-10]
4. Infrastructure Safety: [1-10]
5. Crime Safety: [1-10]
Overall Score: [X]/50
Main Concerns: [List]
Strengths: [List]
```

### 4. **Risk Assessment Prompt** (Risk Categories)
**Purpose**: Identify and categorize safety risks
**Use Case**: Risk analysis, hazard identification
**Output Format**:
```
HIGH RISK: [List]
MEDIUM RISK: [List]
LOW RISK: [List]
Overall Risk Level: [LOW/MEDIUM/HIGH]
Primary Risk Factor: [Main concern]
```

### 5. **Accessibility Prompt** (Accessibility Focus)
**Purpose**: Evaluate accessibility and inclusive design
**Use Case**: Accessibility assessment, inclusive design
**Output Format**:
```
- Wheelchair Access: [1-10]
- Visual Accessibility: [1-10]
- Mobility Device Support: [1-10]
- Clear Pathways: [1-10]
Accessibility Score: [X]/40
Barriers Identified: [List]
Improvements Needed: [Suggestions]
```

## 📊 **Generated Training Data Summary**

- **Total Images**: 30 street view images
- **Total Training Examples**: 90 prompts
- **Prompt Types**: 4 different types per image (randomly selected)
- **File**: `vlm_safety_training_data.json`

## 🔄 **Next Steps for VLM Fine-tuning**

1. **Review Generated Prompts**: Check `vlm_safety_training_data.json`
2. **Annotate Responses**: Add expected responses for each prompt
3. **Choose VLM Model**: Select base model (e.g., LLaVA, BLIP-2, InstructBLIP)
4. **Fine-tune**: Use your annotated dataset for training
5. **Evaluate**: Test on held-out street view images

## 💡 **Tips for Better Results**

- **Diverse Prompts**: Mix different prompt types for robust training
- **Consistent Formatting**: Ensure responses follow the specified format
- **Quality Annotations**: Use expert knowledge or GPT-4 for initial annotations
- **Validation Set**: Keep 20% of images for validation
- **Iterative Improvement**: Refine prompts based on model performance

## 🛠️ **Quick Usage**

```python
from vlm_workflow import VLMSafetyWorkflow

# Initialize workflow
workflow = VLMSafetyWorkflow()

# Generate training data
training_data = workflow.create_training_dataset()

# Save to file
workflow.save_training_data(training_data)
```
