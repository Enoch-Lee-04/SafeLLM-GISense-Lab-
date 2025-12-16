#!/usr/bin/env python3
"""
Generate training data in various formats from versioned manifests.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from safellm_vlm.data import load_manifest, generate_all_formats
from safellm_vlm.data.generators import generate_openai_jsonl, generate_qwen_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate training data from manifests")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "processed",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--format",
        choices=["all", "openai", "qwen"],
        default="all",
        help="Output format(s) to generate",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="Include base64 encoded images (OpenAI only)",
    )
    parser.add_argument(
        "--image-base-path",
        type=Path,
        default=project_root / "data" / "images" / "image",
        help="Base path for resolving image paths",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GENERATING TRAINING DATA FROM MANIFEST")
    print("=" * 60)
    print(f"Manifest: {args.manifest}")
    print(f"Output: {args.output_dir}")
    print(f"Format: {args.format}")
    print()
    
    # Load manifest
    manifest = load_manifest(args.manifest)
    print(f"Loaded manifest: {manifest.version} ({manifest.split})")
    print(f"Items: {len(manifest.items)}")
    print()
    
    # Generate formats
    if args.format == "all":
        results = generate_all_formats(
            manifest,
            args.output_dir,
            include_images=args.include_images,
            image_base_path=args.image_base_path,
        )
        for fmt, count in results.items():
            print(f"[OK] Generated {fmt}: {count} examples")
    
    elif args.format == "openai":
        suffix = "_images" if args.include_images else "_text"
        output_path = args.output_dir / "openai_finetuning" / f"{manifest.version}_{manifest.split}{suffix}.jsonl"
        count = generate_openai_jsonl(
            manifest,
            output_path,
            include_images=args.include_images,
            image_base_path=args.image_base_path,
        )
        print(f"[OK] Generated OpenAI JSONL: {output_path} ({count} examples)")
    
    elif args.format == "qwen":
        output_path = args.output_dir / "qwen_training" / f"{manifest.version}_{manifest.split}.json"
        count = generate_qwen_dataset(
            manifest,
            output_path,
            image_base_path=args.image_base_path,
        )
        print(f"[OK] Generated Qwen dataset: {output_path} ({count} examples)")
    
    print("\n[OK] Generation complete!")


if __name__ == "__main__":
    main()

