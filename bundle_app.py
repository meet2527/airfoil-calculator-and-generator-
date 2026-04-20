import subprocess
import os
import shutil

def build():
    print("Starting build process...")
    
    # Path to your main entry point
    entry_point = "run_app.py"
    
    # PyInstaller command
    # We use --onefile for a single executable
    # We use --noconsole if we don't want a terminal popping up (but for debugging we might keep it)
    # We use --add-data to include the templates folder
    # We use --collect-all for pywebview and fastapi-related libs
    
    cmd = [
        ".venv\\Scripts\\pyinstaller",
        "--noconfirm",
        "--onefile",
        "--windowed", # No console window
        "--name", "RC_Airfoil_Config",
        "--add-data", f"templates{os.pathsep}templates",
        "--collect-all", "webview",
        "--collect-all", "fastapi",
        "--collect-all", "uvicorn",
        "--collect-all", "airfoil_config",
        "--hidden-import", "uvicorn.logging",
        entry_point
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print("\nBuild complete!")
    print("The executable can be found in the 'dist' folder.")

if __name__ == "__main__":
    build()
