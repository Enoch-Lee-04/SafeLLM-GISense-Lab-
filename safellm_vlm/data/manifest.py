"""
Manifest schema validation and utilities for versioned data pipelines.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime


@dataclass
class ManifestItem:
    """A single item in the data manifest."""
    image_path: str
    score: float
    label: str  # e.g., "very_unsafe", "unsafe", "neutral", "safe", "very_safe"
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ManifestItem":
        return cls(
            image_path=data["image_path"],
            score=data["score"],
            label=data["label"],
            metadata=data.get("metadata", {}),
            reasons=data.get("reasons", []),
        )


@dataclass
class ManifestSchema:
    """Schema for versioned data manifests."""
    version: str
    split: str  # "train", "val", "test"
    items: List[ManifestItem]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "split": self.split,
            "created_at": self.created_at,
            "description": self.description,
            "items": [item.to_dict() for item in self.items],
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ManifestSchema":
        items = [ManifestItem.from_dict(item) for item in data.get("items", [])]
        return cls(
            version=data["version"],
            split=data["split"],
            items=items,
            created_at=data.get("created_at", datetime.now().isoformat()),
            description=data.get("description", ""),
        )


# Valid labels for safety scores
VALID_LABELS = {
    "very_unsafe",   # 0-2
    "unsafe",        # 2-4
    "neutral",       # 4-6
    "moderately_safe",  # 6-8
    "safe",          # 8-10
}

VALID_SPLITS = {"train", "val", "test"}


class ManifestValidationError(Exception):
    """Raised when manifest validation fails."""
    pass


def validate_manifest(manifest: ManifestSchema, check_images: bool = False) -> List[str]:
    """
    Validate a manifest against the schema.
    
    Args:
        manifest: The manifest to validate
        check_images: If True, verify image files exist
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    if not manifest.version:
        errors.append("Missing required field: version")
    
    if manifest.split not in VALID_SPLITS:
        errors.append(f"Invalid split '{manifest.split}'. Must be one of: {VALID_SPLITS}")
    
    if not manifest.items:
        errors.append("Manifest has no items")
    
    # Validate each item
    for i, item in enumerate(manifest.items):
        prefix = f"Item {i}"
        
        if not item.image_path:
            errors.append(f"{prefix}: Missing image_path")
        elif check_images and not Path(item.image_path).exists():
            errors.append(f"{prefix}: Image not found: {item.image_path}")
        
        if item.score is None:
            errors.append(f"{prefix}: Missing score")
        elif not (0 <= item.score <= 10):
            errors.append(f"{prefix}: Score {item.score} out of range [0, 10]")
        
        if item.label not in VALID_LABELS:
            errors.append(f"{prefix}: Invalid label '{item.label}'. Must be one of: {VALID_LABELS}")
    
    return errors


def load_manifest(path: Path) -> ManifestSchema:
    """
    Load a manifest from a JSON file.
    
    Args:
        path: Path to the manifest JSON file
        
    Returns:
        Parsed ManifestSchema
        
    Raises:
        ManifestValidationError: If manifest is invalid
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    manifest = ManifestSchema.from_dict(data)
    
    errors = validate_manifest(manifest)
    if errors:
        raise ManifestValidationError(f"Invalid manifest: {errors}")
    
    return manifest


def save_manifest(manifest: ManifestSchema, path: Path) -> None:
    """
    Save a manifest to a JSON file.
    
    Args:
        manifest: The manifest to save
        path: Output path for the JSON file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)


def score_to_label(score: float) -> str:
    """Convert a numeric score (0-10) to a categorical label."""
    if score < 2:
        return "very_unsafe"
    elif score < 4:
        return "unsafe"
    elif score < 6:
        return "neutral"
    elif score < 8:
        return "moderately_safe"
    else:
        return "safe"

