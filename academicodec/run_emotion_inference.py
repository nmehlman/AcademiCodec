import os
import sys
import torch
import json
import tqdm
import torch
import torchaudio
import torch.nn.functional as F

import sys

sys.path.append("/home/nmehlman/emo-steer/vox-profile-release/src/model/emotion")
from wavlm_emotion_dim import WavLMWrapper  # type: ignore
from whisper_emotion_dim import WhisperWrapper # type: ignore
from whisper_emotion import WhisperWrapper as WhisperWrapperCat # type: ignore

import argparse

EMOTION_LABEL_LIST = [
    'Anger', 
    'Contempt', 
    'Disgust', 
    'Fear', 
    'Happiness', 
    'Neutral', 
    'Sadness', 
    'Surprise', 
    'Other'
]

def load_voxprofile_models(device: str = "cpu", categorical: bool = False) -> tuple:
    """
    Load the WavLM and Whisper emotion models.

    Args:
        device: Device string ('cpu' or 'cuda').
        categorical: Whether to use categorical models.
    Returns:
        Tuple containing the WavLM and Whisper models.
    """
    if categorical:
        model = WhisperWrapperCat.from_pretrained("tiantiaf/whisper-large-v3-msp-podcast-emotion").to(device)
        model.eval()
        return model

    else:
        wavlm_model = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-msp-podcast-emotion-dim").to(device)
        wavlm_model.eval()

        whisper_model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-msp-podcast-emotion-dim").to(device)
        whisper_model.eval()
        
        # for param in whisper_model.parameters():
        #     param.data = param.data.to(device)
        #     if param._grad is not None:
        #         param._grad.data = param._grad.data.to(device)

        return wavlm_model, whisper_model

def compute_voxprofile_predictions(audio, wavlm_model, whisper_model, categorical_model = None) -> tuple:
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

        if categorical_model is not None:
            logits, embedding, _, _, _, _ = categorical_model(
                audio, return_feature=True
            )
            emotion_probs = F.softmax(logits, dim=1).squeeze()

            return arousal, valence, emotion_probs

        return arousal, valence


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run emotion inference using VoxProfile models.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio_dir", type=str, help="Path to reference audio directory.")
    group.add_argument("--file_list", type=str, help="Path to file list.")
    parser.add_argument("--output", required=True, help="Output file for emotion labels.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run models on (cpu or cuda).")
    args = parser.parse_args()

    audio_dir = args.audio_dir
    device = args.device

    wavlm_model, whisper_model = load_voxprofile_models(device=device)
    categorical_model = load_voxprofile_models(device=device, categorical=True)

    emotion_labels = {}

    if args.file_list is not None:
        with open(args.file_list, "r") as f:
            audio_files = [line.strip() for line in f.readlines()]
    else:
        audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]

    for file_path in tqdm.tqdm(audio_files):
        try:
            audio, sr = torchaudio.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        
        audio = audio.to(device)
        arousal, valence, emotions_probs = compute_voxprofile_predictions(audio, wavlm_model, whisper_model, categorical_model=categorical_model)
        emotion_labels[file_path.split('/')[-1].split('.')[0]] = {
            "arousal": round(arousal, 4),
            "valence": round(valence, 4),
            "emotions": {
                EMOTION_LABEL_LIST[i]: round(emotions_probs[i].item(), 4) for i in range(len(EMOTION_LABEL_LIST))
            }
        }

    with open(args.output, "w") as f:
        json.dump(emotion_labels, f, indent=4)