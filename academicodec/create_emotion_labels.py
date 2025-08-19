import os
import sys
import torch
import json
import tqdm
import torch
import torchaudio

import sys

# VoxProfile emotion models
sys.path.append("/home/nmehlman/emo-steer/vox-profile-release/src/model/emotion")
from wavlm_emotion_dim import WavLMWrapper
from whisper_emotion_dim import WhisperWrapper

import argparse

def load_voxprofile_models(device: str = "cpu") -> tuple:
    """
    Load the WavLM and Whisper emotion models.

    Args:
        device: Device string ('cpu' or 'cuda').

    Returns:
        Tuple containing the WavLM and Whisper models.
    """
    wavlm_model = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-msp-podcast-emotion-dim").to(device)
    wavlm_model.eval()

    whisper_model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-msp-podcast-emotion-dim").to(device)
    whisper_model.eval()
    
    for param in whisper_model.parameters():
        param.data = param.data.to(device)
        if param._grad is not None:
            param._grad.data = param._grad.data.to(device)

    return wavlm_model, whisper_model

def compute_voxprofile_predictions(audio, wavlm_model, whisper_model) -> tuple:
    """
    Compute average arousal and valence predictions for a list of audio files.

    Args:
        audio: Audio vector.
        wavlm_model: WavLM teacher model.
        whisper_model: Whisper teacher model.
        device: Device string ('cpu' or 'cuda').

    Returns:
        Dictionary with filenames as keys and predictions as values.
    """
    # Ensure audio is on the same device as the models
    model_device = next(wavlm_model.parameters()).device # DEBUG
    audio = audio.to(model_device) # DEBUG
    with torch.no_grad():
        wavlm_arousal, wavlm_valence, _ = wavlm_model(audio)
        whisper_arousal, whisper_valence, _ = whisper_model(audio)

        # Average predictions
        arousal = torch.stack([wavlm_arousal.squeeze(-1), whisper_arousal.squeeze(-1)], dim=0).mean().item()
        valence = torch.stack([wavlm_valence.squeeze(-1), whisper_valence.squeeze(-1)], dim=0).mean().item()

    return arousal, valence


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Prepare emotion data using VoxProfile models.")
    parser.add_argument("--train_lst", type=str, required=True, help="Path to train file list.")
    parser.add_argument("--val_lst", type=str, required=True, help="Path to val file list.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run models on (cpu or cuda).")
    args = parser.parse_args()

    train_lst = args.train_lst
    val_lst = args.val_lst
    device = args.device

    # Output directory for emotion labels (same as file list dir)
    output_dir = os.path.dirname(train_lst)

    # Read file lists
    with open(train_lst, 'r') as f:
        train_files = [l.strip() for l in f]
    with open(val_lst, 'r') as f:
        val_files = [l.strip() for l in f]

    wavlm_model, whisper_model = load_voxprofile_models(device=device)

    for split, file_list in zip(["train", "val"], [train_files, val_files]):
        emotion_labels = {}
        for filename in tqdm.tqdm(file_list, desc=f"Processing {split} set"):
            try:
                audio, sr = torchaudio.load(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
          
            audio = audio.to(device)
            arousal, valence = compute_voxprofile_predictions(audio, wavlm_model, whisper_model)
            emotion_labels[filename] = {
                "arousal": arousal,
                "valence": valence
            }

        # Save emotion labels as <split>_emotion_labs.json in same dir as file list
        out_path = os.path.join(output_dir, f"{split}_emotion_labs.json")
        with open(out_path, "w") as f:
            json.dump(emotion_labels, f, indent=4)