import sys
import os
import logging
import subprocess
import traceback

EXT_PATH = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, EXT_PATH)

log = logging.getLogger("AniDoc")

download_models = True

try:
    folder_paths_path = os.path.abspath(os.path.join(os.path.dirname(EXT_PATH), "..", "..", "folder_paths.py"))

    sys.path.append(os.path.dirname(folder_paths_path))

    import folder_paths
    
    DIFFUSERS_DIR = os.path.join(folder_paths.models_dir, "diffusers")
    ANIDOC_DIR = os.path.join(DIFFUSERS_DIR, "anidoc")
    SVD_I2V_DIR = os.path.join(
        DIFFUSERS_DIR,
        "stable-video-diffusion-img2vid-xt",
    )
except:
    download_models = False

COTRACKER = os.path.join(EXT_PATH, "cotracker")
    
try:
    log.info(f"Installing requirements")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", f"{EXT_PATH}/requirements.txt", "--no-warn-script-location"])
    
    if download_models:
        from huggingface_hub import snapshot_download

        try:
            if not os.path.exists(ANIDOC_DIR):
                log.info(f"Downloading AniDoc model to: {ANIDOC_DIR}")
                snapshot_download(
                    repo_id="Yhmeng1106/anidoc",
                    local_dir=DIFFUSERS_DIR,
                    local_dir_use_symlinks=False,
                )
        except Exception:
            traceback.print_exc()
            log.error(f"Failed to download AniDoc model")
            
        try:
            if not os.path.exists(SVD_I2V_DIR):
                log.info(f"Downloading stable diffusion video img2vid to: {SVD_I2V_DIR}")
                snapshot_download(
                    repo_id="vdo/stable-video-diffusion-img2vid-xt-1-1",
                    allow_patterns=[f"*.json", "*fp16*"],
                    ignore_patterns=["*unet*"],
                    local_dir=SVD_I2V_DIR,
                    local_dir_use_symlinks=False,
                )
        except Exception:
            traceback.print_exc()
            log.error(f"Failed to download stable diffusion video img2vid")
    
    try:
        log.info("Installing CoTracker")
        subprocess.check_call([sys.executable, "-m", "pip", "install", COTRACKER])
    except Exception:
        traceback.print_exc()
        log.error(f"Failed to install CoTracker")
    
    log.info(f"AniDoc Installation completed")
        
except Exception:
    traceback.print_exc()
    log.error(f"AniDoc Installation failed")