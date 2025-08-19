
#!/bin/bash
source path.sh
set -e

pretrained_ckpt="/data2/nmehlman/models/HiFi-Codec/HiFi-Codec-16k-320d"
log_root="/data2/nmehlman/logs/hifi-codec/HiFi-Codec-16k-320d-standard"

# .lst save the wav path.
input_training_file="/home/nmehlman/emo-steer/AcademiCodec/data/expresso/train.lst"
input_validation_file="/home/nmehlman/emo-steer/AcademiCodec/data/expresso/val.lst"

## finetune 
echo "Finetuning model..."
export CUDA_VISIBLE_DEVICES=0,1
/data2/nmehlman/anaconda3/envs/hifi-codec/bin/python ${BIN_DIR}/finetune.py \
--config config_16k_320d.json \
--checkpoint_path ${log_root} \
--input_training_file ${input_training_file} \
--input_validation_file ${input_validation_file} \
--checkpoint_interval 2500 \
--summary_interval 100 \
--validation_interval 2500 \
--training_epochs 250 \
--pretrained_ckpt ${pretrained_ckpt} \

