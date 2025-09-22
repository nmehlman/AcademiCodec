#!/bin/bash
source path.sh

ckpt=/data2/nmehlman/logs/hifi-codec/HiFi-Codec-16k-320d-adv.8/g_00130000
echo checkpoint path: ${ckpt}

# the path of test wave
file_list=/home/nmehlman/emo-steer/AcademiCodec/data/expresso/val.lst

temp=$(dirname ${ckpt})/temp
output_dir=$(dirname ${ckpt})/full-emotion-eval

# Create samples, reference, and generated directories
mkdir -p "${temp}"
mkdir -p "${output_dir}"

# Run generation, saving output to samples/generated
/data2/nmehlman/anaconda3/envs/hifi-codec/bin/python ${BIN_DIR}/vqvae_copy_syn.py \
    --model_path="${ckpt}" \
    --config_path=config_16k_320d.json \
    --audio_file_list="${file_list}" \
    --outputdir="${temp}" \
    --num_gens=10000 \
    --sample_rate=16000

# Run emotion inference
echo "Running emotion inference on generated audio..."
/data2/nmehlman/anaconda3/envs/hifi-codec/bin/python ../../academicodec/run_emotion_inference.py \
    --audio_dir="${temp}" \
    --output="${output_dir}/emotion_labels_gen.json" \
    --device="cuda"

rm -rf "${temp}"

# Run emotion inference
echo "Running emotion inference on reference audio..."
/data2/nmehlman/anaconda3/envs/hifi-codec/bin/python ../../academicodec/run_emotion_inference.py \
    --file_list="${file_list}" \
    --output="${output_dir}/emotion_labels_ref.json" \
    --device="cuda"

/data2/nmehlman/anaconda3/envs/hifi-codec/bin/python ../../academicodec/plot_emotion_results.py "${output_dir}"
