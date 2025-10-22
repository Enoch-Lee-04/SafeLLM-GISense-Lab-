from openai import OpenAI
import os
import json
from pathlib import Path
from datetime import datetime

# Load API key from file
api_key_path = Path(__file__).parent.parent.parent / "API_KEY_Enoch"
with open(api_key_path, "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

# Function to create a file with the Files API
def create_file(file_path):
    with open(file_path, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="vision",
        )
        return result.id

# Function to evaluate a single image
def evaluate_image(image_path):
    print(f"Evaluating: {image_path.name}")
    file_id = create_file(str(image_path))
    
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Analyze this street view image and provide a safety score from 1-10 (10 being safest)."},
                {
                    "type": "input_image",
                    "file_id": file_id,
                    "detail": "high"
                },
            ],
        }],
    )
    
    return {
        "image": image_path.name,
        "response": response.output_text
    }

# Get all images from the folder
image_folder = Path(__file__).parent.parent.parent / "data" / "images" / "image"
image_files = sorted(image_folder.glob("*.jpg"))

print(f"Found {len(image_files)} images to evaluate\n")

# Evaluate all images
results = []
for image_path in image_files:
    try:
        result = evaluate_image(image_path)
        results.append(result)
        print(f"Response: {result['response']}\n")
    except Exception as e:
        print(f"Error evaluating {image_path.name}: {e}\n")
        results.append({
            "image": image_path.name,
            "error": str(e)
        })

# Save results to JSON file
output_path = Path(__file__).parent.parent.parent / "results" / f"chatgpt_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_path}")