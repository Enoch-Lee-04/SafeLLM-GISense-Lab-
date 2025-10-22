#!/usr/bin/env python3
"""
Local deployment script for Soundscape-to-Image generation using CEUS.pt
This script provides an easy way to generate images from audio files.
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate images from soundscapes using CEUS.pt")
    parser.add_argument('--audio-file', type=str, required=True, 
                       help="Path to input audio file (.wav, .mp3, .flac, .m4a)")
    parser.add_argument('--output-dir', type=str, default='./generated_images',
                       help="Directory to save generated images")
    parser.add_argument('--cond-scale', type=float, default=1.0,
                       help="Conditioning scale (higher = more adherence to audio features)")
    parser.add_argument('--ceus-model', type=str, default='../CEUS.pt',
                       help="Path to CEUS.pt model file")
    parser.add_argument('--audio-encoder', type=str, default='../wlc.pt',
                       help="Path to wlc.pt audio encoder")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found!")
        return 1
    
    if not os.path.exists(args.ceus_model):
        print(f"Error: CEUS model '{args.ceus_model}' not found!")
        print("Please ensure CEUS.pt is in the parent directory or specify correct path with --ceus-model")
        return 1
    
    if not os.path.exists(args.audio_encoder):
        print(f"Error: Audio encoder '{args.audio_encoder}' not found!")
        print("Please ensure wlc.pt is in the parent directory or specify correct path with --audio-encoder")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference
    cmd = f"""python inference.py \\
        --audio-encoder-ckpt "{args.audio_encoder}" \\
        --unet-ckpt "{args.ceus_model}" \\
        --test-audio-path "{args.audio_file}" \\
        --test-image-path "{args.output_dir}" \\
        --cond-scale {args.cond_scale}"""
    
    print("Running inference...")
    print(f"Command: {cmd}")
    
    result = os.system(cmd)
    
    if result == 0:
        print(f"\n✅ Success! Generated images saved to: {args.output_dir}")
        
        # List generated files
        generated_files = [f for f in os.listdir(args.output_dir) if f.endswith('.png')]
        if generated_files:
            print("Generated files:")
            for file in generated_files:
                print(f"  - {file}")
    else:
        print("❌ Error occurred during inference")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
