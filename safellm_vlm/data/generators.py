"""
Format generators for training data.

Generates OpenAI fine-tuning JSONL, Qwen training datasets, etc. from manifests.
"""

import json
import base64
import io
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

from .manifest import ManifestSchema, ManifestItem


# System prompt for safety assessment models
SAFETY_SYSTEM_PROMPT = """You are a Vision Language Model trained to evaluate the perceived safety of a single street-view image.

Your task is to analyze one image at a time and predict a safety score from 0 to 10:

1. A **safety score from 0 to 10**, where:
   - 0 means extremely unsafe
   - 10 means extremely safe

2. After providing the score, explain your reasoning in **2–5 concise bullet points**, each grounded only in **visible environmental cues** present in the image. Base your assessment entirely on what is visible — do not infer details beyond the image.

Visual elements to consider:
- Lighting and visibility: streetlights, shadows, time of day, visibility distance
- Maintenance and cleanliness: litter, graffiti, broken infrastructure, general upkeep
- Social activity: presence of people, pedestrians, vehicles, or community engagement
- Environment and design: open vs. enclosed areas, fencing, barriers, accessibility
- Safety features: crosswalks, sidewalks, cameras, signage, traffic control

Respond using this exact format:
Score: X/10
Reason: ["reason 1", "reason 2", "reason 3", ...]

Be specific and consistent. Avoid vague or overly generic explanations. Use the full 0–10 scale when appropriate."""


def _encode_image_base64(image_path: str, max_size: tuple = (2000, 2000)) -> str:
    """Encode an image to base64 string."""
    with Image.open(image_path) as img:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def _format_response(item: ManifestItem) -> str:
    """Format an item into the expected response format."""
    reasons_str = json.dumps(item.reasons) if item.reasons else '["Assessment based on visible features"]'
    return f"Score: {int(item.score)}/10\nReason: {reasons_str}"


def generate_openai_jsonl(
    manifest: ManifestSchema,
    output_path: Path,
    include_images: bool = False,
    image_base_path: Optional[Path] = None,
) -> int:
    """
    Generate OpenAI fine-tuning JSONL from a manifest.
    
    Args:
        manifest: Source data manifest
        output_path: Path to write the JSONL file
        include_images: Whether to include base64 encoded images
        image_base_path: Base path to resolve relative image paths
        
    Returns:
        Number of examples written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in manifest.items:
            image_path = item.image_path
            if image_base_path:
                image_path = str(image_base_path / item.image_path)
            
            image_filename = Path(image_path).name
            user_prompt = f"Analyze this street view image and provide a safety score.\n\nImage: {image_filename}"
            assistant_response = _format_response(item)
            
            if include_images and Path(image_path).exists():
                image_base64 = _encode_image_base64(image_path)
                messages = [
                    {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": image_base64}},
                        ],
                    },
                    {"role": "assistant", "content": assistant_response},
                ]
            else:
                messages = [
                    {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]
            
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1
    
    return count


def generate_qwen_dataset(
    manifest: ManifestSchema,
    output_path: Path,
    image_base_path: Optional[Path] = None,
) -> int:
    """
    Generate Qwen2-VL training dataset from a manifest.
    
    Args:
        manifest: Source data manifest
        output_path: Path to write the dataset JSON file
        image_base_path: Base path to resolve relative image paths
        
    Returns:
        Number of examples written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset = []
    for item in manifest.items:
        image_path = item.image_path
        if image_base_path:
            image_path = str(image_base_path / item.image_path)
        
        image_filename = Path(image_path).name
        user_prompt = f"Analyze this street view image and provide a safety score.\n\nImage: {image_filename}"
        
        example = {
            "image_path": image_path,
            "prompt": user_prompt,
            "task_type": "safety_score",
            "expected_format": 'Score: X/10\nReason: ["reason 1", "reason 2", ...]',
            "expected_response": _format_response(item),
        }
        dataset.append(example)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    return len(dataset)


def generate_all_formats(
    manifest: ManifestSchema,
    output_dir: Path,
    include_images: bool = False,
    image_base_path: Optional[Path] = None,
) -> Dict[str, int]:
    """
    Generate all training formats from a single manifest.
    
    Args:
        manifest: Source data manifest
        output_dir: Directory to write output files
        include_images: Whether to include base64 encoded images in OpenAI format
        image_base_path: Base path to resolve relative image paths
        
    Returns:
        Dict mapping format name to number of examples written
    """
    results = {}
    
    # OpenAI text-only
    openai_text_path = output_dir / "openai_finetuning" / f"{manifest.version}_{manifest.split}_text.jsonl"
    results["openai_text"] = generate_openai_jsonl(
        manifest, openai_text_path, include_images=False, image_base_path=image_base_path
    )
    
    # OpenAI with images
    if include_images:
        openai_images_path = output_dir / "openai_finetuning" / f"{manifest.version}_{manifest.split}_images.jsonl"
        results["openai_images"] = generate_openai_jsonl(
            manifest, openai_images_path, include_images=True, image_base_path=image_base_path
        )
    
    # Qwen format
    qwen_path = output_dir / "qwen_training" / f"{manifest.version}_{manifest.split}.json"
    results["qwen"] = generate_qwen_dataset(manifest, qwen_path, image_base_path=image_base_path)
    
    return results

