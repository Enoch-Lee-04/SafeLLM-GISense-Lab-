#!/usr/bin/env python3
"""
Convert existing training data to the new versioned manifest format.
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from safellm_vlm.data.manifest import (
    ManifestSchema,
    ManifestItem,
    save_manifest,
    validate_manifest,
    score_to_label,
)


def extract_score_and_reasons(response: str) -> tuple:
    """Extract score and reasons from expected_response."""
    score = None
    reasons = []
    
    # Extract score
    score_match = re.search(r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", response, re.IGNORECASE)
    if score_match:
        score = float(score_match.group(1))
    
    # Extract reasons
    reason_match = re.search(r'Reason:\s*\[(.*?)\]', response, re.DOTALL)
    if reason_match:
        try:
            reasons_str = "[" + reason_match.group(1) + "]"
            reasons = json.loads(reasons_str)
        except json.JSONDecodeError:
            # Fallback: split by comma if JSON fails
            reasons = [r.strip().strip('"\'') for r in reason_match.group(1).split(",")]
    
    return score, reasons


def convert_existing_data(input_path: Path, output_dir: Path, version: str = "safety_v1"):
    """Convert existing training data JSON to versioned manifests."""
    
    print(f"Loading existing data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Found {len(data)} items")
    
    # Group by unique images and use safety_score task type
    items_by_image = {}
    for item in data:
        if item.get("task_type") != "safety_score":
            continue
        
        image_path = item.get("image_path", "")
        if not image_path:
            continue
        
        # Use the image filename as key
        image_filename = Path(image_path).name
        
        if image_filename not in items_by_image:
            expected_response = item.get("expected_response", "")
            score, reasons = extract_score_and_reasons(expected_response)
            
            if score is None:
                continue
            
            # Create relative path for manifest
            relative_path = f"images/{image_filename}"
            
            items_by_image[image_filename] = ManifestItem(
                image_path=relative_path,
                score=score,
                label=score_to_label(score),
                metadata={"source": "GSV", "original_path": image_path},
                reasons=reasons,
            )
    
    print(f"Extracted {len(items_by_image)} unique items")
    
    # Split into train/val/test (80/10/10)
    items = list(items_by_image.values())
    n = len(items)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train_items = items[:train_end]
    val_items = items[train_end:val_end]
    test_items = items[val_end:]
    
    # Create manifests
    manifests = {
        "train": ManifestSchema(
            version=version,
            split="train",
            items=train_items,
            description="Training split for street view safety assessment",
        ),
        "val": ManifestSchema(
            version=version,
            split="val",
            items=val_items,
            description="Validation split for street view safety assessment",
        ),
        "test": ManifestSchema(
            version=version,
            split="test",
            items=test_items,
            description="Test split for street view safety assessment",
        ),
    }
    
    # Save manifests
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split, manifest in manifests.items():
        output_path = output_dir / f"{version}_{split}.json"
        
        # Validate before saving
        errors = validate_manifest(manifest)
        if errors:
            print(f"[WARNING] {split} manifest has validation errors: {errors}")
        
        save_manifest(manifest, output_path)
        print(f"[OK] Saved {split} manifest: {output_path} ({len(manifest.items)} items)")
    
    return manifests


def main():
    input_path = project_root / "configs" / "vlm_safety_training_data.json"
    output_dir = project_root / "data" / "annotations"
    
    print("=" * 60)
    print("CONVERTING EXISTING DATA TO VERSIONED MANIFEST FORMAT")
    print("=" * 60)
    
    manifests = convert_existing_data(input_path, output_dir)
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"\nManifests saved to: {output_dir}")
    print("\nTo generate training formats from manifests, use:")
    print("  from safellm_vlm.data import load_manifest, generate_all_formats")
    print("  manifest = load_manifest(Path('data/annotations/safety_v1_train.json'))")
    print("  generate_all_formats(manifest, Path('data/processed'))")


if __name__ == "__main__":
    main()

