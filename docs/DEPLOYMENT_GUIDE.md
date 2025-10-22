# Soundscape-to-Image Deployment Guide

This guide explains how to deploy the CEUS.pt model to generate images from soundscapes.

## Prerequisites

1. **Model Files**: Ensure you have both model files:
   - `CEUS.pt` - The pre-trained Soundscape-to-Image model
   - `wlc.pt` - The audio encoder model

2. **Python Environment**: Python 3.9 or newer

3. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Deployment Options

### Option 1: Local Command-Line Usage (Simplest)

Use the provided deployment script for easy local inference:

```bash
# Basic usage
python deploy_local.py --audio-file path/to/your/audio.wav

# With custom parameters
python deploy_local.py \
    --audio-file path/to/your/audio.wav \
    --output-dir ./my_generated_images \
    --cond-scale 1.5 \
    --ceus-model ../CEUS.pt \
    --audio-encoder ../wlc.pt
```

**Parameters:**
- `--audio-file`: Path to input audio file (.wav, .mp3, .flac, .m4a)
- `--output-dir`: Directory to save generated images (default: ./generated_images)
- `--cond-scale`: Conditioning scale (default: 1.0, higher = more adherence to audio)
- `--ceus-model`: Path to CEUS.pt model file
- `--audio-encoder`: Path to wlc.pt audio encoder

## Usage Examples

### Command Line Examples

```bash
# Generate image from a single audio file
python deploy_local.py --audio-file ./test_audio/example_1.wav

# Generate with higher conditioning scale
python deploy_local.py --audio-file ./test_audio/example_1.wav --cond-scale 2.0

# Use custom model paths
python deploy_local.py \
    --audio-file ./test_audio/example_1.wav \
    --ceus-model /path/to/CEUS.pt \
    --audio-encoder /path/to/wlc.pt
```


## Performance Considerations

1. **GPU Usage**: The model works on both CPU and GPU, but GPU is significantly faster
2. **Memory Requirements**: 
   - CPU: ~4GB RAM minimum
   - GPU: ~6GB VRAM recommended
3. **Processing Time**: 
   - CPU: 30-60 seconds per image
   - GPU: 5-15 seconds per image

## Troubleshooting

### Common Issues

1. **Model files not found:**
   - Ensure CEUS.pt and wlc.pt are in the correct locations
   - Use absolute paths if needed

2. **CUDA out of memory:**
   - Reduce batch size or use CPU
   - Close other GPU-intensive applications

3. **Audio format issues:**
   - Supported formats: .wav, .mp3, .flac, .m4a, .ogg
   - Audio will be automatically resampled to 16kHz

4. **Import errors:**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.9+ required)

### Getting Help

- Check the original README.md for detailed project information
- Verify model files are downloaded correctly
- Ensure audio files are in supported formats

## Production Deployment

For production deployment:

1. **Use GPU-enabled instances** for better performance
2. **Implement proper logging** and monitoring
3. **Set up health checks** and auto-restart policies
4. **Use process managers** like PM2 or systemd for service management
5. **Monitor resource usage** and set up alerts

## Security Considerations

- Validate input audio files
- Sanitize file names and paths
- Set appropriate file size limits
- Use secure file handling practices
