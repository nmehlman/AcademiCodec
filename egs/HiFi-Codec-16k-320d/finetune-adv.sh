
#!/bin/bash
source path.sh
set -e

pretrained_ckpt="/data2/nmehlman/models/HiFi-Codec/HiFi-Codec-16k-320d"
log_root="/data2/nmehlman/logs/hifi-codec/HiFi-Codec-16k-320d-adv.13"

if [ -d "${log_root}" ]; then
    echo "Error: Log directory ${log_root} already exists."
    exit 1
fi

# .lst save the wav path.
input_training_file="/home/nmehlman/emo-steer/AcademiCodec/data/expresso/train.lst"
training_emotion_files="/home/nmehlman/emo-steer/AcademiCodec/data/expresso/emotion_labels.json"
input_validation_file="/home/nmehlman/emo-steer/AcademiCodec/data/expresso/val.lst"
validation_emotion_files="/home/nmehlman/emo-steer/AcademiCodec/data/expresso/emotion_labels.json"

## finetune 
echo "Finetuning model..."
export CUDA_VISIBLE_DEVICES=0
/data2/nmehlman/anaconda3/envs/hifi-codec/bin/python ${BIN_DIR}/finetune_adv.py \
--config config_16k_320d.json \
--checkpoint_path ${log_root} \
--input_training_file ${input_training_file} \
--input_validation_file ${input_validation_file} \
--training_emotion_labels ${training_emotion_files} \
--validation_emotion_labels ${validation_emotion_files} \
--checkpoint_interval 2000 \
--summary_interval 100 \
--validation_interval 2000 \
--training_epochs 200 \
--pretrained_ckpt ${pretrained_ckpt} \
