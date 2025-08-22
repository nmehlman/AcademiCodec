#!/bin/bash
source path.sh

ckpt=/data2/nmehlman/logs/hifi-codec/HiFi-Codec-16k-320d-adv.12/g_00166000
echo checkpoint path: ${ckpt}

# the path of test wave
wav_dir=/data2/nmehlman/logs/hifi-codec/test_samples

samples_dir=$(dirname ${ckpt})/samples
reference_dir=${samples_dir}/reference
generated_dir=${samples_dir}/generated

# Create samples, reference, and generated directories
mkdir -p "${reference_dir}" "${generated_dir}"

# Copy wav_dir to samples/reference
cp -r "${wav_dir}"/* "${reference_dir}/"

# Run generation, saving output to samples/generated
/data2/nmehlman/anaconda3/envs/hifi-codec/bin/python ${BIN_DIR}/vqvae_copy_syn.py \
    --model_path="${ckpt}" \
    --config_path=config_16k_320d.json \
    --input_wavdir="${wav_dir}" \
    --outputdir="${generated_dir}" \
    --num_gens=10000 \
    --sample_rate=16000

# Run emotion inference
echo "Running emotion inference on generated audio..."
/data2/nmehlman/anaconda3/envs/hifi-codec/bin/python ../../academicodec/run_emotion_inference.py \
    --audio_dir="${generated_dir}" \
    --output="${samples_dir}/emotion_labels_gen.json" \
    --device="cuda"

# Run emotion inference
echo "Running emotion inference on reference audio..."
/data2/nmehlman/anaconda3/envs/hifi-codec/bin/python ../../academicodec/run_emotion_inference.py \
    --audio_dir="${reference_dir}" \
    --output="${samples_dir}/emotion_labels_ref.json" \
    --device="cuda"

/data2/nmehlman/anaconda3/envs/hifi-codec/bin/python ../../academicodec/plot_emotion_results.py "${samples_dir}"

