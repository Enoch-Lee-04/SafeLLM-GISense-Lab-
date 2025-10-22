#!/usr/bin/env python3
"""
Re-run evaluation with improved generation parameters
Fixes repetition and incomplete response issues
"""

import os
import json
import torch
import re
from pathlib import Path
from PIL import Image
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


def load_model(model_path):
    """Load the fine-tuned Qwen2-VL model"""
    print(f"Loading model from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    print("Model loaded successfully!")
    return model, processor


def clean_response(response, prompt_text):
    """
    Clean up model response to remove:
    - Repeated prompts
    - Template echoing
    - Incomplete sentences
    """
    
    # Remove the input prompt if it's echoed
    prompt_clean = prompt_text.replace("<|image_pad|>", "").strip()
    if prompt_clean and prompt_clean in response:
        response = response.split(prompt_clean)[-1].strip()
    
    # Remove common template patterns and instructional text
    template_patterns = [
        r'\[1-10\]',
        r'\[SAFE/UNSAFE\]',
        r'\[HIGH/MEDIUM/LOW\]',
        r'\[Brief explanation.*?\]',
        r'\[List.*?elements\]',
        r'\[Main safety concern\]',
        r'\[.*?explanation.*?\]',
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
    
    for pattern in template_patterns:
        response = re.sub(pattern, '', response, flags=re.IGNORECASE)
    
    # Detect and remove repetitive patterns and rambling
    lines = response.split('\n')
    seen_lines = set()
    clean_lines = []
    repetition_count = 0
    
    # Stop phrases that indicate rambling
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
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Check for rambling/conversational phrases
        line_lower = line_stripped.lower()
        if any(phrase in line_lower for phrase in stop_phrases):
            break
            
        # Check if line is repeating
        if line_stripped in seen_lines:
            repetition_count += 1
            if repetition_count >= 2:
                break
        else:
            repetition_count = 0
            seen_lines.add(line_stripped)
            clean_lines.append(line_stripped)
    
    response = '\n'.join(clean_lines)
    
    # Remove incomplete sentences at the end
    if response.endswith(' ['):
        response = response[:response.rfind(' [')].strip()
    
    # If response contains multiple "Safety Score:" or similar, keep only first
    score_matches = list(re.finditer(r'Safety Score:', response))
    if len(score_matches) > 1:
        # Keep content up to second occurrence
        response = response[:score_matches[1].start()].strip()
    
    # Remove trailing incomplete patterns
    response = re.sub(r'\s+\w+$', lambda m: '' if len(m.group().strip()) < 3 else m.group(), response)
    
    return response.strip()


def extract_safety_score(response):
    """Extract numerical safety score from response - comprehensive version"""
    
    # Remove markdown image syntax that can interfere
    response_clean = re.sub(r'!\[.*?\]\(.*?\)', '', response)
    response_clean = re.sub(r'\[.*?Image.*?\]', '', response_clean)
    
    # Try to find explicit score patterns (most specific first)
    score_patterns = [
        # Direct score formats
        r'Safety Score:\s*\*?\*?(\d+(?:\.\d+)?)\*?\*?',  # "Safety Score: **6**" or "Safety Score: 6"
        r'Score:\s*\*?\*?(\d+(?:\.\d+)?)\*?\*?',  # "Score: **6**"
        r'safety score of\s*(\d+(?:\.\d+)?)',
        r'safety score:\s*(\d+(?:\.\d+)?)',
        r'score is\s*(\d+(?:\.\d+)?)',
        r'assign.*?score.*?(\d+(?:\.\d+)?)',
        r'rate.*?(\d+(?:\.\d+)?)\s*(?:out of|/)\s*10',
        r'(?:Overall|Final)?\s*Score:\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*/\s*10',  # "6/10" format
        r'scored?\s*(?:at|as)?\s*(\d+(?:\.\d+)?)',
        # Reasoning patterns
        r'Reasoning:.*?(\d+(?:\.\d+)?)',
        r'reasoning.*?score.*?(\d+(?:\.\d+)?)',
        # Bullet point patterns
        r'\*\s*Reasonable\s*-.*?(\d+(?:\.\d+)?)',
        # "Would be" patterns - extract the number mentioned
        r'would be.*?(\d+(?:\.\d+)?)',
        r'example.*?(\d+(?:\.\d+)?)',
    ]
    
    for pattern in score_patterns:
        matches = list(re.finditer(pattern, response_clean, re.IGNORECASE))
        for match in matches:
            try:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    # Extra validation: avoid "Scored Out" type phrases
                    context_before = response_clean[max(0, match.start()-20):match.start()].lower()
                    if 'scored out' in context_before or 'score out' in context_before:
                        continue
                    return score
            except (ValueError, IndexError):
                continue
    
    # More aggressive: look for any number 0-10 in safety context
    # Split into sentences and analyze each
    sentences = re.split(r'[.!?\n]+', response_clean)
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Skip sentences that are clearly descriptive/hypothetical
        if any(skip in sentence_lower for skip in ['would be', 'example score', 'scored out', 'could be']):
            # But still check if there's an actual score given
            pass
        
        # Look for numbers with safety context in this sentence
        if any(word in sentence_lower for word in ['safe', 'score', 'rate', 'risk', 'assess', 'reason']):
            number_matches = list(re.finditer(r'\b([0-9]|10)(?:\.\d+)?\b', sentence))
            for match in number_matches:
                try:
                    score = float(match.group(1))
                    if 0 <= score <= 10:
                        # Avoid false positives (years, lane numbers, etc.)
                        if any(avoid in sentence_lower for avoid in ['lane', 'floor', 'year', '201', '202']):
                            continue
                        return score
                except (ValueError, IndexError):
                    continue
    
    # Fallback: Try to infer from risk level descriptions
    if re.search(r'HIGH\s+RISK', response, re.IGNORECASE) or re.search(r'high risk', response, re.IGNORECASE):
        return 3.0
    elif re.search(r'MEDIUM\s+RISK', response, re.IGNORECASE) or re.search(r'medium risk', response, re.IGNORECASE):
        return 5.0
    elif re.search(r'LOW\s+RISK', response, re.IGNORECASE) or re.search(r'low risk', response, re.IGNORECASE):
        return 7.0
    
    # Check for explicit SAFE/UNSAFE classification
    if re.search(r'Classification:\s*SAFE\b', response, re.IGNORECASE):
        return 7.0
    elif re.search(r'Classification:\s*UNSAFE\b', response, re.IGNORECASE):
        return 3.0
    
    # Check for descriptive safety terms
    if re.search(r'\b(very\s+)?safe\b', response, re.IGNORECASE) and not re.search(r'unsafe', response, re.IGNORECASE):
        return 7.0
    elif re.search(r'\bunsafe\b', response, re.IGNORECASE):
        return 3.0
    
    # Check for reasonable/unreasonable mentions
    if re.search(r'\breasonable\b', response, re.IGNORECASE):
        return 6.0
    
    # Last resort: look for descriptive safety indicators
    positive_indicators = ['well-lit', 'clean', 'clear', 'good', 'adequate', 'maintained', 'safe', 'no.*obstacles', 'no.*hazards']
    negative_indicators = ['poor', 'dark', 'unsafe', 'danger', 'hazard', 'obstacles', 'cracks', 'potholes', 'debris']
    
    response_lower = response.lower()
    pos_count = sum(1 for indicator in positive_indicators if re.search(indicator, response_lower))
    neg_count = sum(1 for indicator in negative_indicators if re.search(indicator, response_lower))
    
    # If response discusses safety but has no explicit score
    if 'road' in response_lower or 'street' in response_lower or 'safe' in response_lower:
        if pos_count > neg_count and pos_count >= 1:
            return 6.5  # Reasonably safe based on description
        elif neg_count > pos_count and neg_count >= 1:
            return 3.5  # Less safe based on description
        elif 'clear' in response_lower or 'no.*visible' in response_lower:
            return 6.0  # Default for neutral/clear descriptions
    
    return None


def generate_safety_assessment(model, processor, image_path, prompt_type="score"):
    """Generate safety assessment with improved parameters"""
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Very explicit prompts that demand specific format
    prompts = {
        "score": "<|image_pad|>Analyze this street view image and provide a safety score from 1-10 (10 being safest). Format: 'Safety Score: [number]\\nReasoning: [brief explanation]'",
        
        "binary": "<|image_pad|>Classify this street view as SAFE or UNSAFE for pedestrians. Format: 'Classification: [SAFE/UNSAFE]\\nConfidence: [HIGH/MEDIUM/LOW]\\nReason: [brief explanation]'",
        
        "detailed": "<|image_pad|>Provide detailed safety analysis. Evaluate: 1.Pedestrian Safety: ___ 2.Traffic Safety: ___ 3.Lighting Safety: ___ 4.Infrastructure Safety: ___ 5.Crime Safety: ___ Overall Score: ___/50 Main Concerns: [top 3] Strengths: [top 3]",
        
        "risks": "<|image_pad|>Assess safety risks by category. Format: 'HIGH RISK: [elements]\\nMEDIUM RISK: [elements]\\nLOW RISK: [elements]\\nOverall Risk Level: [LOW/MEDIUM/HIGH]\\nPrimary Risk Factor: [concern]'"
    }
    
    text = prompts.get(prompt_type, prompts["score"])
    
    try:
        # Process inputs
        inputs = processor(text=[text], images=[image], return_tensors='pt')
        
        # Generate with improved parameters to prevent rambling
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,      # Strict limit to prevent rambling
                min_new_tokens=10,       # Ensure minimal response
                do_sample=False,         # Greedy decoding for consistency
                temperature=1.0,         # Not used with greedy
                num_beams=1,             # Single beam for efficiency
                repetition_penalty=1.8,  # Strong penalty for repetition
                no_repeat_ngram_size=5,  # Prevent 5-gram repetition
                length_penalty=0.8,      # Prefer shorter, focused responses
                early_stopping=True,     # Stop at EOS
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        # Decode response
        full_response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Clean the response
        response = clean_response(full_response, text)
        
        # Extract score if possible
        score = extract_safety_score(response)
        
        return response, score
        
    except Exception as e:
        return f"Error: {str(e)}", None


def evaluate_all_images(model, processor, image_folder, output_excel):
    """Evaluate model on all images with improved generation"""
    
    print(f"\nEvaluating fine-tuned model on: {image_folder}")
    
    # Get all images
    image_folder = Path(image_folder)
    image_files = sorted(list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png")))
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    print(f"Found {len(image_files)} images\n")
    
    # Results storage
    all_results = []
    
    # Process each image
    for image_file in tqdm(image_files, desc="Evaluating images"):
        image_name = image_file.stem
        
        # Generate assessment with score prompt (most important)
        response, score = generate_safety_assessment(
            model, processor, str(image_file), "score"
        )

        # Binary classification
        binary_response, _ = generate_safety_assessment(
            model, processor, str(image_file), "binary"
        )

        # Store results
        all_results.append({
            'Image ID': image_name,
            'Image Path': str(image_file),
            'Safety Score': score if score else "N/A",
            'Score Response': response,
            'Binary Classification': binary_response,
            'Response Length': len(response),
            'Has Score': "Yes" if score else "No",
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Load ground truth if available
    try:
        gt_file = Path("configs/vlm_safety_training_data.json")
        if gt_file.exists():
            with open(gt_file, 'r') as f:
                training_data = json.load(f)
            
            # Create ground truth mapping
            gt_mapping = {}
            for item in training_data:
                img_id = Path(item['image_path']).stem
                # Extract score from expected response
                score_match = re.search(r'Safety Score:\s*(\d+)', item['expected_response'])
                if score_match:
                    gt_mapping[img_id] = int(score_match.group(1))
            
            # Add ground truth column
            df['Ground Truth'] = df['Image ID'].map(gt_mapping)
            
            # Calculate error for valid scores
            df['Absolute Error'] = df.apply(
                lambda row: abs(row['Safety Score'] - row['Ground Truth']) 
                if pd.notna(row['Ground Truth']) and isinstance(row['Safety Score'], (int, float))
                else None,
                axis=1
            )
    except Exception as e:
        print(f"Note: Could not load ground truth: {e}")
    
    # Save to Excel
    print(f"\nSaving results to: {output_excel}")
    
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # Main results
        df.to_excel(writer, sheet_name='Evaluation Results', index=False)
        
        # Summary statistics
        stats_data = {
            'Metric': [
                'Total Images',
                'Successful Scores',
                'Failed Scores',
                'Average Score',
                'Average Response Length',
                'Evaluation Date'
            ],
            'Value': [
                len(df),
                df['Has Score'].value_counts().get('Yes', 0),
                df['Has Score'].value_counts().get('No', 0),
                f"{df['Safety Score'].apply(lambda x: x if isinstance(x, (int, float)) else None).mean():.2f}",
                f"{df['Response Length'].mean():.1f}",
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        
        if 'Absolute Error' in df.columns:
            valid_errors = df['Absolute Error'].dropna()
            if len(valid_errors) > 0:
                stats_data['Metric'].extend([
                    'Mean Absolute Error',
                    'Median Absolute Error',
                    'Images with Ground Truth'
                ])
                stats_data['Value'].extend([
                    f"{valid_errors.mean():.2f}",
                    f"{valid_errors.median():.2f}",
                    len(valid_errors)
                ])
        
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_excel(writer, sheet_name='Statistics', index=False)
        
        # Problem cases (no score extracted)
        problem_df = df[df['Has Score'] == 'No'][['Image ID', 'Score Response']]
        if len(problem_df) > 0:
            problem_df.to_excel(writer, sheet_name='Problem Cases', index=False)
    
    print(f"\n{'='*70}")
    print(f"Evaluation complete!")
    print(f"{'='*70}")
    print(f"\nResults:")
    print(f"  - Total images: {len(df)}")
    print(f"  - Successful scores: {df['Has Score'].value_counts().get('Yes', 0)}")
    print(f"  - Failed scores: {df['Has Score'].value_counts().get('No', 0)}")
    
    if 'Absolute Error' in df.columns and df['Absolute Error'].notna().any():
        print(f"  - Mean Absolute Error: {df['Absolute Error'].mean():.2f}")
    
    print(f"\nResults saved to: {output_excel}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Re-evaluate model with improved generation')
    parser.add_argument('--model_path', type=str, default='models/fine_tuned_qwen2vl_trained',
                        help='Path to fine-tuned model')
    parser.add_argument('--image_folder', type=str, default='data/images/image',
                        help='Folder containing images')
    parser.add_argument('--output', type=str, default='results/fine_tuned_evaluation_improved.xlsx',
                        help='Output Excel file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Improved Model Evaluation")
    print("="*70)
    
    # Load model
    model, processor = load_model(args.model_path)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Evaluate
    evaluate_all_images(model, processor, args.image_folder, args.output)


if __name__ == "__main__":
    main()

