import torch
import torchaudio
import os
import jiwer
from jiwer import transforms as tr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from academicodec.models.hificodec.vqvae_tester import VqvaeTester


if __name__ == "__main__":

    data_dir = "/data1/open_data/LibriSpeech/dev-clean/"
    config_path = "/home/nmehlman/emo-steer/AcademiCodec/egs/HiFi-Codec-16k-320d/config_16k_320d.json"
    ckpt_path = "/data2/nmehlman/logs/hifi-codec/HiFi-Codec-16k-320d-adv.8/g_00130000"
    model_id = "openai/whisper-large-v3"
    device = "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load HiFiCodec model
    model = VqvaeTester(config_path=config_path, model_path=ckpt_path, sample_rate=16000)
    model.to(device)
    model.vqvae.generator.remove_weight_norm()
    model.vqvae.encoder.remove_weight_norm()
    model.eval()

    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    asr_model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    transform = tr.Compose([
        tr.ToLowerCase(),
        tr.RemovePunctuation(),
        tr.Strip(),
        tr.RemoveMultipleSpaces(),
        tr.ReduceToListOfListOfWords() 
    ])

    all_wer_ref, all_wer_gen = [], []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        
        if len(filenames) == 0:
            continue

        spk, sess = dirpath.split("/")[-2:]
        transcripts_path = os.path.join(dirpath, f"{spk}-{sess}.trans.txt")

        for line in open(transcripts_path):
            
            utt_id, reference = line.strip().split(" ", 1)

            audio_path = os.path.join(dirpath, f"{utt_id}.flac")

            x_raw, sr = torchaudio.load(audio_path)
            assert sr == 16000

            # Run conversion
            _, x_gen = model(audio_path, device=device)
            x_gen = x_gen.squeeze().numpy()

            result_raw = pipe(x_raw.squeeze().numpy())
            hypothesis = result_raw["text"] # type: ignore

            result_gen = pipe(x_gen)
            hypothesis = result_gen["text"] # type: ignore

            wer_raw = jiwer.wer(reference, hypothesis, reference_transform = transform, hypothesis_transform = transform) # type: ignore
            wer_gen = jiwer.wer(reference, hypothesis, reference_transform = transform, hypothesis_transform = transform) # type: ignore

            all_wer_ref.append(wer_raw)
            all_wer_gen.append(wer_gen)

    print(f"Avg WER (raw): {sum(all_wer_ref)/len(all_wer_ref)}")
    print(f"Avg WER (gen): {sum(all_wer_gen)/len(all_wer_gen)}")