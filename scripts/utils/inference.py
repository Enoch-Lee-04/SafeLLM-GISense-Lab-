import torch
import argparse
import os
import numpy as np
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from torchvggish import vggish
from PIL import Image
import soundfile as sf
from torchvision import transforms

def load_audio_model(audio_encoder_path, device):
    """Load the pre-trained audio encoder model"""
    model = vggish.WLC(urls="", pretrained=False).to(device)
    model.load_state_dict(torch.load(audio_encoder_path, weights_only=False, map_location=device).state_dict())
    model.eval()
    return model

def load_imagen_model(unet_ckpt_path, device):
    """Load the trained Imagen model"""
    # Create the same Unet architecture as in training
    unet1 = Unet(
        dim=128,
        cond_dim=512,
        dim_mults=(1, 2, 4, 8),
        num_resnet_blocks=3,
        layer_attns=(False, True, True, True),
    )

    unet2 = Unet(
        dim=128,
        cond_dim=512,
        dim_mults=(1, 2, 4, 8),
        num_resnet_blocks=(2, 4, 8, 8),
        layer_attns=(False, False, False, True),
        layer_cross_attns=(False, False, False, True)
    )

    imagen = Imagen(
        unets=(unet1, unet2),
        image_sizes=(64, 256),
        timesteps=256,
        cond_drop_prob=0.1,
        lowres_sample_noise_level=0.1,
        random_crop_sizes=(None, 64)
    )

    trainer = ImagenTrainer(imagen).to(device)
    trainer.load(unet_ckpt_path)
    
    return trainer.imagen

def generate_image_from_audio(audio_path, audio_model, imagen_model, device, cond_scale=1.0):
    """Generate an image from an audio file"""
    # Load and process audio
    audio_data, sample_rate = sf.read(audio_path)
    
    # Ensure audio is mono and has the right sample rate
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Resample if necessary (VGGish expects 16kHz)
    if sample_rate != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
    
    # Get audio embedding
    with torch.no_grad():
        audio_embedding = audio_model(audio_data, 16000)["embedding"]
        audio_embedding = audio_embedding.to(device)
    
    # Generate image
    with torch.no_grad():
        generated_images = imagen_model.sample(
            batch_size=1,
            text_embeds=audio_embedding,
            cond_scale=cond_scale,
            return_pil_images=True
        )
    
    return generated_images[0]

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    print("Loading audio encoder...")
    audio_model = load_audio_model(args.audio_encoder_ckpt, device)
    
    print("Loading Imagen model...")
    imagen_model = load_imagen_model(args.unet_ckpt, device)
    
    # Create output directory
    os.makedirs(args.test_image_path, exist_ok=True)
    
    # Process audio files
    if os.path.isfile(args.test_audio_path):
        # Single audio file
        audio_files = [args.test_audio_path]
    else:
        # Directory of audio files
        audio_files = [os.path.join(args.test_audio_path, f) 
                      for f in os.listdir(args.test_audio_path) 
                      if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
    
    print(f"Found {len(audio_files)} audio files to process")
    
    for audio_file in audio_files:
        print(f"Processing: {audio_file}")
        
        try:
            # Generate image
            generated_image = generate_image_from_audio(
                audio_file, 
                audio_model, 
                imagen_model, 
                device, 
                args.cond_scale
            )
            
            # Save image
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_path = os.path.join(args.test_image_path, f"{base_name}_generated.png")
            generated_image.save(output_path)
            print(f"Generated image saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from audio using trained Imagen model")
    parser.add_argument('--audio-encoder-ckpt', type=str, required=True, 
                       help="Path to the pre-trained audio encoder checkpoint")
    parser.add_argument('--unet-ckpt', type=str, required=True,
                       help="Path to the trained Unet checkpoint")
    parser.add_argument('--test-audio-path', type=str, required=True,
                       help="Path to test audio file or directory")
    parser.add_argument('--test-image-path', type=str, required=True,
                       help="Path to save generated images")
    parser.add_argument('--cond-scale', type=float, default=1.0,
                       help="Conditioning scale for image generation")
    
    args = parser.parse_args()
    main(args)
