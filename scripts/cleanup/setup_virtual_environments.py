#!/usr/bin/env python3
"""
Virtual Environment Setup Script for Soundscape-to-Image Project
This script creates separate virtual environments for different project components
to optimize space usage and dependency management.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {command}")
            print(f"Error output: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running command {command}: {e}")
        return False

def create_virtual_environment(venv_name, venv_path):
    """Create a virtual environment"""
    print(f"\nCreating virtual environment: {venv_name}")
    print(f"   Location: {venv_path}")
    
    if os.path.exists(venv_path):
        print(f"   INFO: Virtual environment already exists at {venv_path}")
        print(f"   Using existing environment")
        return True
    
    # Create virtual environment
    if not run_command(f"python -m venv \"{venv_path}\""):
        print(f"   FAILED: Could not create virtual environment")
        return False
    
    print(f"   SUCCESS: Virtual environment created successfully")
    return True

def install_requirements(venv_path, requirements_file, venv_name):
    """Install requirements in virtual environment"""
    if not os.path.exists(requirements_file):
        print(f"   WARNING: Requirements file not found: {requirements_file}")
        return True
    
    print(f"   Installing requirements from {requirements_file}")
    
    # Determine the correct pip path based on OS
    if platform.system() == "Windows":
        pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
        python_path = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        pip_path = os.path.join(venv_path, "bin", "pip")
        python_path = os.path.join(venv_path, "bin", "python")
    
    # Upgrade pip first
    if not run_command(f"\"{python_path}\" -m pip install --upgrade pip"):
        print(f"   WARNING: Failed to upgrade pip, continuing anyway...")
    
    # Install requirements
    if not run_command(f"\"{pip_path}\" install -r \"{requirements_file}\""):
        print(f"   FAILED: Could not install requirements for {venv_name}")
        return False
    
    print(f"   SUCCESS: Requirements installed successfully")
    return True

def create_activation_script(venv_name, venv_path, project_dir):
    """Create activation script for easy environment switching"""
    script_name = f"activate_{venv_name}.bat" if platform.system() == "Windows" else f"activate_{venv_name}.sh"
    script_path = os.path.join(project_dir, script_name)
    
    if platform.system() == "Windows":
        script_content = f"""@echo off
echo Activating {venv_name} virtual environment...
call "{venv_path}\\Scripts\\activate.bat"
echo.
echo ✅ {venv_name} environment activated!
echo 💡 To deactivate, run: deactivate
echo.
"""
    else:
        script_content = f"""#!/bin/bash
echo "Activating {venv_name} virtual environment..."
source "{venv_path}/bin/activate"
echo ""
echo "✅ {venv_name} environment activated!"
echo "💡 To deactivate, run: deactivate"
echo ""
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    if platform.system() != "Windows":
        os.chmod(script_path, 0o755)
    
    print(f"   Created activation script: {script_name}")

def main():
    """Main setup function"""
    print("Setting up Virtual Environments for Soundscape-to-Image Project")
    print("=" * 70)
    
    # Get project directory
    project_dir = Path(__file__).parent.absolute()
    print(f"Project directory: {project_dir}")
    
    # Define virtual environments
    environments = [
        {
            "name": "main",
            "path": os.path.join(project_dir, "venv_main"),
            "requirements": "requirements.txt",
            "description": "Main project dependencies (torch, librosa, opencv, etc.)"
        },
        {
            "name": "qwen",
            "path": os.path.join(project_dir, "venv_qwen"),
            "requirements": "qwen_requirements.txt",
            "description": "Qwen VLM fine-tuning dependencies"
        },
        {
            "name": "visual",
            "path": os.path.join(project_dir, "venv_visual"),
            "requirements": "visual_safety_requirements.txt",
            "description": "Visual safety assessment dependencies"
        }
    ]
    
    # Create virtual environments
    success_count = 0
    for env in environments:
        print(f"\n{'='*50}")
        print(f"Setting up: {env['name'].upper()} Environment")
        print(f"Description: {env['description']}")
        print(f"{'='*50}")
        
        # Create virtual environment
        if create_virtual_environment(env['name'], env['path']):
            # Install requirements
            requirements_path = os.path.join(project_dir, env['requirements'])
            if install_requirements(env['path'], requirements_path, env['name']):
                # Create activation script
                create_activation_script(env['name'], env['path'], project_dir)
                success_count += 1
                print(f"   SUCCESS: {env['name'].upper()} environment setup complete!")
            else:
                print(f"   FAILED: {env['name'].upper()} environment setup failed!")
        else:
            print(f"   FAILED: {env['name'].upper()} environment setup failed!")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SETUP SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully created: {success_count}/{len(environments)} environments")
    
    if success_count > 0:
        print(f"\nNEXT STEPS:")
        print(f"1. Activate an environment using the activation scripts:")
        for env in environments:
            script_name = f"activate_{env['name']}.bat" if platform.system() == "Windows" else f"activate_{env['name']}.sh"
            print(f"   - {script_name} (for {env['description']})")
        
        print(f"\n2. Or manually activate:")
        for env in environments:
            if platform.system() == "Windows":
                print(f"   - {env['name']}: {env['path']}\\Scripts\\activate.bat")
            else:
                print(f"   - {env['name']}: source {env['path']}/bin/activate")
        
        print(f"\nSPACE SAVINGS:")
        print(f"   - Isolated dependencies per project")
        print(f"   - Easy cleanup by deleting venv folders")
        print(f"   - No global package accumulation")
        
        print(f"\nVirtual environments created in:")
        for env in environments:
            print(f"   - {env['name']}: {env['path']}")
    
    print(f"\nSetup complete!")

if __name__ == "__main__":
    main()
