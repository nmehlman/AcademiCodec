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
from whisper_emotion import WhisperWrapper
import torch.nn.functional as F

import argparse

def load_voxprofile_models(device: str = "cpu"):
    model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-msp-podcast-emotion").to(device)
    model.eval()
    return model

def compute_voxprofile_predictions(audio, model):
    logits, embedding, _, _, _, _ = model(
        audio, return_feature=True
    )
    emotion_probs = F.softmax(logits, dim=1).squeeze()
    return emotion_probs.cpu()

if __name__ == "__main__":

    train_lst = "TODO"
    val_lst = "TODO"
    output_dir = "TODO"
    device = "cpu"

    # Read file lists
    with open(train_lst, 'r') as f:
        train_files = [l.strip() for l in f]
    with open(val_lst, 'r') as f:
        val_files = [l.strip() for l in f]

    model = load_voxprofile_models(device=device)

    for split, file_list in zip(["train", "val"], [train_files, val_files]):
        emotion_labels = {}
        for filename in tqdm.tqdm(file_list, desc=f"Processing {split} set"):
            try:
                audio, sr = torchaudio.load(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
          
            audio = audio.to(device)
            preds = compute_voxprofile_predictions(audio, model)
            emotion_labels[filename] = preds.tolist()
        

        # Save emotion labels as <split>_emotion_labs.json in same dir as file list
        out_path = os.path.join(output_dir, f"{split}_emotion_labs_cat.json")
        with open(out_path, "w") as f:
            json.dump(emotion_labels, f, indent=4)