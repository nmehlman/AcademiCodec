
#!/bin/bash
source path.sh
set -e

pretrained_ckpt="/data1/nmehlman/models/HiFi-Codec/HiFi-Codec-16k-320d"
log_root="/data1/nmehlman/logs/hifi-codec/test"

if [ -d "${log_root}" ]; then
    echo "Error: Log directory ${log_root} already exists."
    exit 1
fi

# .lst save the wav path.
input_training_file="/data1/nmehlman/data/expresso-parsed/train.lst"
emotion_file="/data1/nmehlman/data/expresso-parsed/emotion_labels.json"
input_validation_file="/data1/nmehlman/data/expresso-parsed/val.lst"

## finetune 
echo "Finetuning model..."
export CUDA_VISIBLE_DEVICES=0
/data1/nmehlman/miniconda3/envs/hifi-codec/bin/python ${BIN_DIR}/finetune_adv.py \
--config config_16k_320d.json \
--checkpoint_path ${log_root} \
--input_training_file ${input_training_file} \
--input_validation_file ${input_validation_file} \
--emotion_labels ${emotion_file} \
--checkpoint_interval 2000 \
--summary_interval 100 \
--validation_interval 2000 \
--training_epochs 200 \
--pretrained_ckpt ${pretrained_ckpt} \
