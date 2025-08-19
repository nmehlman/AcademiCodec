import os
import sys
import torch
import json
import tqdm
import torch
import torchaudio

import sys
sys.path.append("/home/nmehlman/emo-steer/vox-profile-release/src/model/emotion")
from wavlm_emotion_dim import WavLMWrapper  # type: ignore
from whisper_emotion_dim import WhisperWrapper # type: ignore

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
    with torch.no_grad():
        wavlm_arousal, wavlm_valence, _ = wavlm_model(audio)
        whisper_arousal, whisper_valence, _ = whisper_model(audio)

        # Average predictions
        arousal = torch.stack([wavlm_arousal.squeeze(-1), whisper_arousal.squeeze(-1)], dim=0).mean().item()
        valence = torch.stack([wavlm_valence.squeeze(-1), whisper_valence.squeeze(-1)], dim=0).mean().item()

    return arousal, valence


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run emotion inference using VoxProfile models.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to reference audio directory.")
    parser.add_argument("--output", required=True, help="Output file for emotion labels.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run models on (cpu or cuda).")
    args = parser.parse_args()

    audio_dir = args.audio_dir
    device = args.device

    wavlm_model, whisper_model = load_voxprofile_models(device=device)

    emotion_labels = {}
    for filename in tqdm.tqdm(os.listdir(audio_dir)):
        if not filename.endswith(".wav"): continue
        file_path = os.path.join(audio_dir, filename)
        try:
            audio, sr = torchaudio.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        
        audio = audio.to(device)
        arousal, valence = compute_voxprofile_predictions(audio, wavlm_model, whisper_model)
        emotion_labels[filename] = {
            "arousal": round(arousal, 4),
            "valence": round(valence, 4)
        }

    with open(args.output, "w") as f:
        json.dump(emotion_labels, f, indent=4)