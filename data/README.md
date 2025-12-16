# Data Directory

This directory contains the versioned data pipeline for street view safety assessment.

## Structure

```
data/
├── data_version.json    # Version tracking and manifest registry
├── raw/                 # Original, unprocessed data
│   ├── images/          # Street view images
│   └── audio/           # Audio files (for future use)
├── annotations/         # Versioned data manifests
│   ├── safety_v1_train.json
│   ├── safety_v1_val.json
│   └── safety_v1_test.json
├── processed/           # Generated training formats
│   ├── openai_finetuning/
│   └── qwen_training/
└── images/              # (Legacy) Original image location
```

## Manifest Format

Each manifest follows a standardized schema:

```json
{
  "version": "safety_v1",
  "split": "train",
  "created_at": "2025-12-05T...",
  "description": "Training split for street view safety assessment",
  "items": [
    {
      "image_path": "images/1.jpg",
      "score": 7.0,
      "label": "moderately_safe",
      "metadata": {"city": "Austin", "source": "GSV"},
      "reasons": ["reason 1", "reason 2"]
    }
  ]
}
```

### Labels

| Score Range | Label |
|-------------|-------|
| 0-2 | very_unsafe |
| 2-4 | unsafe |
| 4-6 | neutral |
| 6-8 | moderately_safe |
| 8-10 | safe |

## Usage

### Load a manifest

```python
from safellm_vlm.data import load_manifest

manifest = load_manifest(Path("data/annotations/safety_v1_train.json"))
print(f"Items: {len(manifest.items)}")
```

### Generate training data

```python
from safellm_vlm.data import load_manifest, generate_all_formats

manifest = load_manifest(Path("data/annotations/safety_v1_train.json"))
results = generate_all_formats(
    manifest,
    output_dir=Path("data/processed"),
    image_base_path=Path("data/raw")
)
```

### CLI Usage

```bash
# Convert existing data to manifest format
python scripts/data/convert_to_manifest.py

# Generate training data from manifest
python scripts/data/generate_training_data.py \
    --manifest data/annotations/safety_v1_train.json \
    --format all
```

## Versioning

When adding new data:

1. Update `data_version.json` with new version entry
2. Create new manifest files with incremented version (e.g., `safety_v2`)
3. Document changes in the changelog

This ensures experiments are reproducible by referencing specific data versions.
