# code based on https://github.com/b04901014/MQTTS
import math
import os
import random

import librosa
import numpy as np
import json
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn


def load_wav(full_path, sr):
    wav, sr = librosa.load(full_path, sr=sr)
    return wav, sr


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y,
                    n_fft,
                    num_mels,
                    sampling_rate,
                    hop_size,
                    win_size,
                    fmin,
                    fmax,
                    center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + '_' +
                  str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int(
            (n_fft - hop_size) / 2)),
        mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True)

    spec = torch.abs(spec)
    #spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r') as f:
        training_files = [l.strip() for l in f]
    with open(a.input_validation_file, 'r') as f:
        validation_files = [l.strip() for l in f]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 training_files,
                 segment_size,
                 n_fft,
                 num_mels,
                 hop_size,
                 win_size,
                 sampling_rate,
                 fmin,
                 fmax,
                 split=True,
                 shuffle=True,
                 n_cache_reuse=1,
                 device=None,
                 fmax_loss=None,
                 fine_tuning=False,
                 base_mels_path=None,
                 emotion_labels=None):
                 
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        if emotion_labels is not None:
            self.emotion_labels = json.load(open(emotion_labels, 'r'))
        else:
            self.emotion_labels = None

    def __getitem__(self, index):
        filename = self.audio_files[index]
        
        if self._cache_ref_count == 0:
            try:
                # Note by yuantian: load with the sample_rate of config
                audio, sampling_rate = load_wav(filename, sr=self.sampling_rate)
            except Exception as e:
                print(f"Error on audio: {filename}")
                audio = np.random.normal(size=(160000, )) * 0.05
                sampling_rate = self.sampling_rate
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start +
                                  self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (
                        0, self.segment_size - audio.size(1)), 'constant')

            mel = mel_spectrogram(
                audio,
                self.n_fft,
                self.num_mels,
                self.sampling_rate,
                self.hop_size,
                self.win_size,
                self.fmin,
                self.fmax,
                center=False)
        else:
            mel = np.load(
                os.path.join(self.base_mels_path,
                             os.path.splitext(os.path.split(filename)[-1])[0] +
                             '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0,
                                               mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(
                        mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (
                        0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (
                        0, self.segment_size - audio.size(1)), 'constant')

        mel_loss = mel_spectrogram(
            audio,
            self.n_fft,
            self.num_mels,
            self.sampling_rate,
            self.hop_size,
            self.win_size,
            self.fmin,
            self.fmax_loss,
            center=False)
        
        if self.emotion_labels is not None:
            emotion_labels = self.emotion_labels[filename.split('/')[-1].split('.')[0]]
            return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze(), emotion_labels)
        else:
            return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    training_files, validation_files = get_dataset_filelist(
        argparse.Namespace(
            input_training_file="/home/nmehlman/emo-steer/AcademiCodec/data/expresso/train.lst",
            input_validation_file="/home/nmehlman/emo-steer/AcademiCodec/data/expresso/val.lst"
        )
    )
    dataset = MelDataset(training_files, segment_size=16000, n_fft=1024,
                         num_mels=80, hop_size=256, win_size=1024,
                         sampling_rate=22050, fmin=0.0, fmax=8000.0,
                         split=True, shuffle=True,
                         emotion_labels="/home/nmehlman/emo-steer/AcademiCodec/data/expresso/emotion_labels.json"
                         )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))

    print("Mel batch shape:", batch[0].shape)
    print("Audio batch shape:", batch[1].shape)
    print("Filename batch:", batch[2])
    print("Mel loss batch shape:", batch[3].shape)
    if dataset.emotion_labels is not None:
        print("Emotion labels batch:", batch[4])
