import os
import argparse
import random

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
    parser = argparse.ArgumentParser(description="Generate .lst files for dataset splits.")
    parser.add_argument("audio_dir", help="Directory containing audio files")
    parser.add_argument("--train_lst", default="train.lst", help="Output training .lst file")
    parser.add_argument("--val_lst", default="val.lst", help="Output validation .lst file")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    audio_files = find_audio_files(args.audio_dir)
    if not audio_files:
        print("No audio files found in the specified directory.")
        return

    random.seed(args.seed)
    random.shuffle(audio_files)
    val_count = int(len(audio_files) * args.val_ratio)
    val_files = audio_files[:val_count]
    train_files = audio_files[val_count:]

    write_list(train_files, args.train_lst)
    write_list(val_files, args.val_lst)

    print(f"Found {len(audio_files)} audio files.")
    print(f"Training files: {len(train_files)} written to {args.train_lst}")
    print(f"Validation files: {len(val_files)} written to {args.val_lst}")

if __name__ == "__main__":
    main()