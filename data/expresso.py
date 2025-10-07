"""
ExpressoDataset: A PyTorch Dataset for the Expresso speech corpus.

This dataset handles both conversational and read speech from the Expresso corpus,
with specialized processing for Voice Activity Detection (VAD) segments in conversational audio.

Key Features:
- Supports both conversational and read speech styles
- Creates separate samples for each VAD segment in conversational audio
- Handles multi-channel conversational audio (channels 1 and 2)
- Deterministic audio processing (no random cropping)
- Configurable resampling and target audio length
- Style filtering to exclude specific speech styles

Dataset Structure:
- Read speech: One sample per file entry
- Conversational speech: One sample per VAD segment per channel
  This significantly increases the dataset size as each VAD interval becomes a separate training sample

Audio Processing:
- Resamples audio from 48kHz to specified rate (default: 16kHz)
- For long audio: takes first target_length_s seconds (no random cropping)
- For short audio: pads with zeros to reach target length
- VAD segments are extracted precisely using start/end timestamps

Usage:
    dataset = ExpressoDataset(
        data_dir='/path/to/expresso/',
        split='train',
        resample_rate=16000,
        filter_styles=['singing', 'whisper'],
        target_length_s=5
    )
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import re
import torchaudio


EXPRESSO_SR = 16000 # Assumed resampled audio, original is 48kHz


class ExpressoDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        resample_rate: int = 16000,
        filter_styles: list = ["singing", "whisper"],
        target_length_s: int = 5,
        audio_dir: str = "audio_16khz",
    ):

        self.data_dir = data_dir
        self.split = split
        self.resample_rate = resample_rate
        self.target_length_s = target_length_s
        self.audio_dir = audio_dir

        splits_file_path = os.path.join(data_dir, "splits", f"{split}.txt")
        self.split_files = open(splits_file_path, "r").readlines()[1:]

        # Apply filtering based on style
        filtered_files = []
        for file in self.split_files:
            style = file.strip().split("\t")[0].split("_")[1]
            if style not in filter_styles:
                filtered_files.append(file)
        self.split_files = filtered_files

        # Parse VAD segments file into a dict: {name: [(start, end), ...]}
        vad_segments = {}
        vad_path = os.path.join(data_dir, "VAD_segments.txt")
        with open(vad_path, "r") as f:
            for line in f.readlines()[3:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                name = parts[0]
                segments = []
                matches = re.findall(r"\(([^,]+),\s*([^)]+)\)", line)
                for start_str, end_str in matches:
                    try:
                        start = float(start_str)
                        end = float(end_str)
                        segments.append((start, end))
                    except ValueError:
                        continue
                vad_segments[name] = segments
        self.vad_segments = vad_segments

        if resample_rate != EXPRESSO_SR:
            self.resample = torchaudio.transforms.Resample(EXPRESSO_SR, resample_rate)
        else:
            self.resample = None

        # Build comprehensive sample index including VAD segments
        self.sample_index = []
        for file_info in self.split_files:
            fname = file_info.strip().split("\t")[0]
            spk, style, id = fname.split("_")[:3]

            if "-" in spk:  # Conversation style
                # Add separate entries for each VAD segment in each channel
                for channel in [1, 2]:
                    vad_key = f"{fname}/channel{channel}"
                    if vad_key in self.vad_segments:
                        segments = self.vad_segments[vad_key]
                        for segment_idx, (start_time, end_time) in enumerate(segments):
                            self.sample_index.append(
                                {
                                    "file_info": file_info,
                                    "fname": fname,
                                    "spk": spk,
                                    "style": style,
                                    "id": id,
                                    "is_convo": True,
                                    "channel": channel,
                                    "vad_segment": (start_time, end_time),
                                    "segment_idx": segment_idx,
                                }
                            )
                    else:
                        # If no VAD data, add one entry per channel with full audio
                        self.sample_index.append(
                            {
                                "file_info": file_info,
                                "fname": fname,
                                "spk": spk,
                                "style": style,
                                "id": id,
                                "is_convo": True,
                                "channel": channel,
                                "vad_segment": None,
                                "segment_idx": None,
                            }
                        )
            else:  # Read style
                self.sample_index.append(
                    {
                        "file_info": file_info,
                        "fname": fname,
                        "spk": spk,
                        "style": style,
                        "id": id,
                        "is_convo": False,
                        "channel": None,
                        "vad_segment": None,
                        "segment_idx": None,
                    }
                )

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):

        sample_info = self.sample_index[idx]
        fname = sample_info["fname"]
        spk = sample_info["spk"]
        style = sample_info["style"]
        id = sample_info["id"]
        is_convo = sample_info["is_convo"]

        if is_convo:
            audio_path = os.path.join(
                self.data_dir,
                self.audio_dir,
                "conversational",
                spk,
                style,
                f"{fname}.wav",
            )
        else:
            if 'longform' in fname:
                    audio_path = os.path.join(
                    self.data_dir,
                    self.audio_dir,
                    "read",
                    spk,
                    style,
                    "longform",
                    f"{fname}.wav",
                )
            else:
                audio_path = os.path.join(
                    self.data_dir,
                    self.audio_dir,
                    "read",
                    spk,
                    style,
                    "base",
                    f"{fname}.wav",
                )

        try:
            audio, sr = torchaudio.load(audio_path)
        except RuntimeError:
            print(f"Audio file not found: {audio_path}")
            audio = torch.zeros(1, 16000)

        if self.resample is not None:
            audio = self.resample(audio)

        # Handle conversational speech with specific VAD segment
        if is_convo:
            channel = sample_info["channel"]
            vad_segment = sample_info["vad_segment"]

            if vad_segment is not None:
                start_time_seg, end_time = vad_segment

                # Convert time to sample indices
                start_sample = int(start_time_seg * self.resample_rate)
                end_sample = int(end_time * self.resample_rate)

                # Extract the segment from the specified channel
                start_sample = max(0, start_sample)
                end_sample = min(audio.size(1), end_sample)
                audio = audio[
                    channel - 1 : channel, start_sample:end_sample
                ]  # Select specific channel
                filename = f"{fname}_channel{channel}_seg{sample_info['segment_idx']}"
            else:
                # If no VAD segment, use the specified channel and full audio
                audio = audio[channel - 1 : channel, :]
                filename = f"{fname}_channel{channel}"


        # Handle read speech with existing start/end parsing (if present)
        elif not is_convo:
            finfo = sample_info["file_info"].strip().split("\t")
            if len(finfo) > 1:
                start_end = finfo[1].strip()
                if start_end and "," in start_end:
                    start_str, end_str = start_end.split(",")
                    try:
                        start = float(start_str.strip("()"))
                        end = float(end_str.strip("()"))

                        start_sample = int(start * self.resample_rate)
                        end_sample = int(end * self.resample_rate)

                        start_sample = max(0, start_sample)
                        end_sample = min(audio.size(1), end_sample)
                        audio = audio[:, start_sample:end_sample]
                    except ValueError:
                        pass  # Use full audio if parsing fails
                    
            filename = fname  # Use filename without extension

        # Pad or truncate audio to target length (always take first target_length_s seconds)
        target_length_samples = int(self.target_length_s * self.resample_rate)
        audio_length = audio.size(1)

        if audio_length < target_length_samples:
            padding = target_length_samples - audio_length
            audio = torch.nn.functional.pad(audio, (0, padding))
            length = audio_length

        elif audio_length >= target_length_samples:
            # Take the first target_length_s seconds instead of random cropping
            audio = audio[:, :target_length_samples]
            length = target_length_samples

        return {
            "audio": audio,
            "speaker": spk,
            "style": style,
            "id": id,
            "length": length,
            "filename": filename,
        }


if __name__ == "__main__":

    import tqdm

    # Example usage
    dataset = ExpressoDataset(
        data_dir="/data1/open_data/expresso/", split="train", resample_rate=16000
    )
    
    print(len(dataset), len(dataset.sample_index))
    
    print(dataset[23653])
