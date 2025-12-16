#!/usr/bin/env python3
"""
Fine-tune GPT-4o-mini on street view safety assessment task
Uses OpenAI's fine-tuning API
"""

import os
import time
import json
from pathlib import Path
from openai import OpenAI
from datetime import datetime

class GPT4oMiniFineTuner:
    """Handle GPT-4o-mini fine-tuning process"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the fine-tuner
        
        Args:
            api_key: OpenAI API key (if None, loads from API_KEY_Enoch file)
        """
        if api_key is None:
            api_key_path = Path(__file__).parent.parent.parent / "API_KEY_Enoch"
            with open(api_key_path, "r") as f:
                api_key = f.read().strip()
        
        self.client = OpenAI(api_key=api_key)
        self.project_root = Path(__file__).parent.parent.parent
        
    def upload_training_file(self, file_path: Path) -> str:
        """
        Upload training file to OpenAI
        
        Args:
            file_path: Path to JSONL training file
            
        Returns:
            File ID from OpenAI
        """
        print(f"[UPLOAD] Uploading training file: {file_path}")
        
        with open(file_path, "rb") as f:
            response = self.client.files.create(
                file=f,
                purpose="fine-tune"
            )
        
        file_id = response.id
        print(f"[OK] File uploaded successfully! File ID: {file_id}")
        
        return file_id
    
    def upload_validation_file(self, file_path: Path) -> str:
        """
        Upload validation file to OpenAI
        
        Args:
            file_path: Path to JSONL validation file
            
        Returns:
            File ID from OpenAI
        """
        print(f"[UPLOAD] Uploading validation file: {file_path}")
        
        with open(file_path, "rb") as f:
            response = self.client.files.create(
                file=f,
                purpose="fine-tune"
            )
        
        file_id = response.id
        print(f"[OK] Validation file uploaded successfully! File ID: {file_id}")
        
        return file_id
    
    def create_fine_tune_job(
        self, 
        training_file_id: str,
        validation_file_id: str = None,
        model: str = "gpt-4o-mini-2024-07-18",
        suffix: str = None,
        hyperparameters: dict = None
    ) -> str:
        """
        Create a fine-tuning job
        
        Args:
            training_file_id: ID of uploaded training file
            validation_file_id: ID of uploaded validation file (optional)
            model: Base model to fine-tune
            suffix: Custom suffix for the fine-tuned model name
            hyperparameters: Training hyperparameters
            
        Returns:
            Fine-tune job ID
        """
        print(f"\n[START] Creating fine-tuning job...")
        print(f"   Base model: {model}")
        
        if suffix is None:
            suffix = f"safety-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Default hyperparameters
        if hyperparameters is None:
            hyperparameters = {
                "n_epochs": 3,  # Number of epochs
                "batch_size": "auto",  # Automatic batch size
                "learning_rate_multiplier": "auto"  # Automatic learning rate
            }
        
        print(f"   Hyperparameters: {hyperparameters}")
        print(f"   Model suffix: {suffix}")
        
        # Create fine-tuning job
        job_params = {
            "training_file": training_file_id,
            "model": model,
            "suffix": suffix,
            "hyperparameters": hyperparameters
        }
        
        if validation_file_id:
            job_params["validation_file"] = validation_file_id
        
        response = self.client.fine_tuning.jobs.create(**job_params)
        
        job_id = response.id
        print(f"[OK] Fine-tuning job created! Job ID: {job_id}")
        
        return job_id
    
    def monitor_fine_tune_job(self, job_id: str, check_interval: int = 60):
        """
        Monitor fine-tuning job progress
        
        Args:
            job_id: Fine-tune job ID
            check_interval: Seconds between status checks
        """
        print(f"\n[MONITOR] Monitoring fine-tuning job: {job_id}")
        print(f"   (Checking every {check_interval} seconds)")
        print("\nThis may take several hours. You can safely stop this script")
        print("and check status later with: check_job_status(job_id)")
        print("-" * 60)
        
        while True:
            try:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                status = job.status
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Status: {status}")
                
                if status == "succeeded":
                    print("\n[SUCCESS] Fine-tuning completed successfully!")
                    print(f"[OK] Fine-tuned model: {job.fine_tuned_model}")
                    return job.fine_tuned_model
                
                elif status == "failed":
                    print("\n[ERROR] Fine-tuning failed!")
                    print(f"Error: {job.error}")
                    return None
                
                elif status == "cancelled":
                    print("\n[WARNING] Fine-tuning was cancelled")
                    return None
                
                # Show progress if available
                if hasattr(job, 'trained_tokens') and job.trained_tokens:
                    print(f"   Trained tokens: {job.trained_tokens}")
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print("\n\n[WARNING] Monitoring stopped (job continues running)")
                print(f"Job ID: {job_id}")
                print("Check status later with:")
                print(f"  python -c \"from finetune_gpt4o_mini import GPT4oMiniFineTuner; ft = GPT4oMiniFineTuner(); ft.check_job_status('{job_id}')\"")
                return None
            
            except Exception as e:
                print(f"\n[ERROR] Error monitoring job: {e}")
                time.sleep(check_interval)
    
    def check_job_status(self, job_id: str):
        """
        Check the status of a fine-tuning job
        
        Args:
            job_id: Fine-tune job ID
        """
        print(f"[STATUS] Checking status of job: {job_id}")
        
        try:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            
            print(f"\nStatus: {job.status}")
            print(f"Model: {job.model}")
            print(f"Created at: {job.created_at}")
            
            if job.finished_at:
                print(f"Finished at: {job.finished_at}")
            
            if job.fine_tuned_model:
                print(f"Fine-tuned model: {job.fine_tuned_model}")
            
            if hasattr(job, 'trained_tokens') and job.trained_tokens:
                print(f"Trained tokens: {job.trained_tokens}")
            
            if job.error:
                print(f"Error: {job.error}")
            
            return job
            
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            return None
    
    def list_fine_tune_jobs(self, limit: int = 10):
        """
        List recent fine-tuning jobs
        
        Args:
            limit: Number of jobs to list
        """
        print(f"📋 Listing {limit} most recent fine-tuning jobs...")
        
        try:
            jobs = self.client.fine_tuning.jobs.list(limit=limit)
            
            print(f"\nFound {len(jobs.data)} jobs:\n")
            
            for i, job in enumerate(jobs.data, 1):
                print(f"{i}. Job ID: {job.id}")
                print(f"   Status: {job.status}")
                print(f"   Model: {job.model}")
                if job.fine_tuned_model:
                    print(f"   Fine-tuned model: {job.fine_tuned_model}")
                print()
            
            return jobs.data
            
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            return []
    
    def save_job_info(self, job_id: str, model_name: str = None):
        """
        Save fine-tuning job information to file
        
        Args:
            job_id: Fine-tune job ID
            model_name: Name of the fine-tuned model
        """
        output_dir = self.project_root / "models" / "gpt4o_mini_finetuned"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        info = {
            "job_id": job_id,
            "model_name": model_name,
            "created_at": datetime.now().isoformat(),
            "base_model": "gpt-4o-mini-2024-07-18"
        }
        
        output_file = output_dir / "fine_tune_info.json"
        
        with open(output_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"[SAVED] Job info saved to: {output_file}")


def main():
    """Main function to run fine-tuning"""
    
    print("=" * 60)
    print("GPT-4o-mini FINE-TUNING FOR SAFETY ASSESSMENT")
    print("=" * 60)
    
    # Initialize fine-tuner
    finetuner = GPT4oMiniFineTuner()
    
    # Data paths
    data_dir = finetuner.project_root / "data" / "openai_finetuning"
    
    # Check which version to use    
    print("\n[FILES] Available training data:")
    print("1. With images (base64 encoded) - train_with_images.jsonl")
    print("2. Text-only - train_text_only.jsonl")
    
    # For now, use text-only as it's more reliable
    # Vision fine-tuning might have additional requirements
    print("\n[INFO] Using text-only version for better reliability")
    
    train_file = data_dir / "train_text_only.jsonl"
    val_file = data_dir / "val_text_only.jsonl"
    
    if not train_file.exists():
        print(f"\n[ERROR] Training file not found: {train_file}")
        print("Please run: python scripts/training/prepare_openai_finetuning_data.py")
        return
    
    # Upload files
    print("\n" + "=" * 60)
    print("STEP 1: UPLOADING FILES")
    print("=" * 60)
    
    training_file_id = finetuner.upload_training_file(train_file)
    
    validation_file_id = None
    if val_file.exists():
        validation_file_id = finetuner.upload_validation_file(val_file)
    
    # Create fine-tuning job
    print("\n" + "=" * 60)
    print("STEP 2: CREATING FINE-TUNING JOB")
    print("=" * 60)
    
    # Hyperparameters tuned for safety assessment
    hyperparameters = {
        "n_epochs": 3,  # Can adjust based on results
        "batch_size": "auto",
        "learning_rate_multiplier": "auto"
    }
    
    job_id = finetuner.create_fine_tune_job(
        training_file_id=training_file_id,
        validation_file_id=validation_file_id,
        hyperparameters=hyperparameters
    )
    
    # Save job info
    finetuner.save_job_info(job_id)
    
    # Monitor job
    print("\n" + "=" * 60)
    print("STEP 3: MONITORING FINE-TUNING PROGRESS")
    print("=" * 60)
    
    model_name = finetuner.monitor_fine_tune_job(job_id, check_interval=60)
    
    if model_name:
        # Update saved info with model name
        finetuner.save_job_info(job_id, model_name)
        
        print("\n" + "=" * 60)
        print("[OK] FINE-TUNING COMPLETE!")
        print("=" * 60)
        print(f"\nYour fine-tuned model: {model_name}")
        print("\nNext steps:")
        print("1. Test the model: python scripts/evaluation/test_gpt4o_mini_finetuned.py")
        print("2. Run comparison: python scripts/evaluation/compare_all_models.py")
    else:
        print("\n[WARNING] Fine-tuning job status unknown")
        print(f"Job ID: {job_id}")
        print("Check status manually with:")
        print(f"  finetuner.check_job_status('{job_id}')")


if __name__ == "__main__":
    main()


