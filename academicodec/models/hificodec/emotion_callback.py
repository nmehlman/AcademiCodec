import numpy as np
import sys
import torch
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("/home/nmehlman/emo-steer/vox-profile-release/src/model/emotion")
from wavlm_emotion_dim import WavLMWrapper  # type: ignore
from whisper_emotion_dim import WhisperWrapper # type: ignore

import io
from PIL import Image

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
        audio = audio.to(next(wavlm_model.parameters()).device)  # Ensure audio is on the same device as the models
        wavlm_arousal, wavlm_valence, _ = wavlm_model(audio)
        whisper_arousal, whisper_valence, _ = whisper_model(audio)

        # Average predictions
        arousal = torch.stack([wavlm_arousal.squeeze(-1), whisper_arousal.squeeze(-1)], dim=0).mean().item()
        valence = torch.stack([wavlm_valence.squeeze(-1), whisper_valence.squeeze(-1)], dim=0).mean().item()

    return {"arousal": arousal, "valence": valence}

def emotion_callback(encoder, quantizer, generator, validation_loader, sw, steps: int = 0, n_batches: int = 1, device: str = "cpu"):
    
    wavlm_model, whisper_model = load_voxprofile_models(device=device)
    
    raw_samples = []
    recon_samples = []
    for j, batch in enumerate(validation_loader):
        
        # Run inference using HiFi-Codec model
        with torch.no_grad():
            x, y, _, y_mel = batch[:4]
            c = encoder(y.to(device).unsqueeze(1))
            q, loss_q, c = quantizer(c)
            y_g_hat = generator(q)
        
        for sample in y:
            raw_samples.append(sample)
        for sample in y_g_hat:
            recon_samples.append(sample)

        if j >= n_batches:
            break

    emotion_preds_raw = [compute_voxprofile_predictions(sample.squeeze().unsqueeze(0), wavlm_model, whisper_model) for sample in raw_samples]
    emotion_preds_recon = [compute_voxprofile_predictions(sample.squeeze().unsqueeze(0), wavlm_model, whisper_model) for sample in recon_samples]

    ref_arousal = [pred["arousal"] for pred in emotion_preds_raw]
    ref_valence = [pred["valence"] for pred in emotion_preds_raw]
    gen_arousal = [pred["arousal"] for pred in emotion_preds_recon]
    gen_valence = [pred["valence"] for pred in emotion_preds_recon]

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=ref_valence, y=ref_arousal, color='blue', label='Reference')
    sns.scatterplot(x=gen_valence, y=gen_arousal, color='red', label='Generated')

    # Draw arrows between ref and gen points for each file
    for i in range(len(ref_valence)):
        plt.arrow(ref_valence[i], ref_arousal[i],
                  gen_valence[i] - ref_valence[i],
                  gen_arousal[i] - ref_arousal[i],
                  color='gray', alpha=0.5, length_includes_head=True, head_width=0.01)
        
    plt.axhline(0.5, color='red', linestyle='--', linewidth=1)
    plt.axvline(0.5, color='red', linestyle='--', linewidth=1)

    plt.xlabel("Valence", fontsize=18)
    plt.ylabel("Arousal", fontsize=18)
    plt.legend(fontsize=16)
    plt.title("Emotion Results: Reference vs Generated", fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    sw.add_image("Emotion Results", image[0], global_step=steps)
    buf.close()
    plt.close()

    