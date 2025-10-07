import os
import argparse
import random
from unittest import skip
import torch
import sys
from pathlib import Path
import json
import torchaudio
import tqdm

from expresso import ExpressoDataset

# VoxProfile emotion models
sys.path.append("/home/nmehlman/emo-steer/vox-profile-release/src/model/emotion")
from wavlm_emotion_dim import WavLMWrapper
from whisper_emotion_dim import WhisperWrapper

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

def find_audio_files(directory, exts=('.wav', '.flac', '.mp3', '.ogg')):
    audio_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(exts):
                audio_files.append(os.path.join(root, f))
    return audio_files

def write_list(filelist, out_path):
    with open(out_path, 'w') as f:
        for item in filelist:
            f.write(item + '\n')

def main():

    data_root = "/data1/open_data/expresso/"
    data_save_path = "/data1/nmehlman/data/expresso-parsed"
    split = "train"
    device = 'cuda:0'
    seed = 42
    val_ratio = 0.05
    
    dataset = ExpressoDataset(
        data_dir=data_root, split=split
    )

    wavlm_model, whisper_model = load_voxprofile_models(device=device)

    emotion_preds = {}

    # Iterate over samples
    all_data_paths = []
    for sample in tqdm.tqdm(dataset, desc=f"Processing {split} set"):

        audio = sample["audio"]
        file_name = sample["filename"]
            
        audio = audio.to(device)
            
        arousal, valence = compute_voxprofile_predictions(audio, wavlm_model, whisper_model)
        assert file_name not in emotion_preds, f"Duplicate file name found: {file_name}"
        emotion_preds[file_name] = {
            "arousal": arousal,
            "valence": valence
        }

        save_path = os.path.join(data_save_path, split, f"{file_name}.wav")
        torchaudio.save(save_path, audio.cpu(), 16000)
        all_data_paths.append(save_path)

    with open(os.path.join(data_save_path, f"emotion_labels.json"), "w") as f: # Save emotion predictions
        json.dump(emotion_preds, f, indent=4)

    # Create train/val lst files
    random.seed(seed)
    random.shuffle(all_data_paths)
    val_count = int(len(all_data_paths) * val_ratio)
    val_files = all_data_paths[:val_count]
    train_files = all_data_paths[val_count:]

    write_list(train_files, os.path.join(data_save_path, f"train_{split}.lst"))
    write_list(val_files, os.path.join(data_save_path, f"val_{split}.lst"))

    print(f"Found {len(all_data_paths)} audio files.")
    print(f"Training files: {len(train_files)} written to {data_save_path}/train.lst")
    print(f"Validation files: {len(val_files)} written to {data_save_path}/val.lst")

if __name__ == "__main__":
    main()