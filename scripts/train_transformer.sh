#!/bin/bash

# Set experiment name and configuration
EXP_NAME="transformer_${CONFIG}_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="./checkpoints/${EXP_NAME}"
SRC_LANG=en
TGT_LANG=it

# Create directories
mkdir -p ${CHECKPOINT_DIR}

# Start training
echo "Starting training for OpusBooks Translation task"
python main.py \
  --src_vocab_size 100 \
  --tgt_vocab_size 100 \
  --src_seq_len 350 \
  --tgt_seq_len 350 \
  --src_lang ${SRC_LANG} \
  --tgt_lang ${TGT_LANG} \
  --d_model 512 \
  --dim_feedforward 2048 \
  --enc_layers 6 \
  --src_tokenizer_path ./tokenizer/${SRC_LANG}.json \
  --tgt_tokenizer_path ./tokenizer/${TGT_LANG}.json \
  --checkpoint_dir ${CHECKPOINT_DIR} \
  --lr 1e-6 \
  --batch_size 8 \
  --weight_decay 1e-4 \
  --epochs 20 \
  --device cpu \

#   | tee ${LOG_DIR}/${EXP_NAME}.log

echo "Training completed. Results saved in ${CHECKPOINT_DIR}"