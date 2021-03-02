#!/bin/bash
TRAIN_CSV="$HOME/Speech/mix-gtt-train-time.csv"
TRAIN_CSV="$HOME/Speech/gtt-35hrs.csv"
#TRAIN_CSV="$HOME/Speech/Tamil/test.csv"
TEST_CSV="$HOME/Speech/mix-gtt-test.csv"

VAL_CSV="$HOME/Speech/gtt-val-35hrs.csv"
TIMESTEP=12
BATCH_SIZE=16
ALPHABETFILE=/shared/kgcoe-research/mil/SpeechCorpora/LibriSpeech_preprocess/english_alphabet.txt
TAMIL_ALPHABET="$HOME/Speech/Tamil/ta-alphabet.txt"
ALPHABET="$HOME/Speech/gtt-alphabet.txt"
#ALPHABET="$HOME/Speech/Gujarati/gu-alphabet.txt"
PHONEME_DICT="$HOME/Speech/labels.pkl"
LEXICON="$HOME/Speech/lexicon.pkl"

MODEL_SAVE_PATH=../../train_models/resnet_v2_gtt_d_6_w_9_wj_4_bd_128_org_spectogram_200_test.pth

LM_PATH="$HOME/Speech/Tamil/ta-lm.binary"
PREVIOUS_MODEL=../../train_models/resnet_v2_phoneme_gtt_d_6_w_9_wj_4_bd_128_org_spectogram.pth

#Original lr: 0.0003 -- tf: 0.00005
#export CUDA_VISIBLE_DEVICES=0
python3 -W ignore ../../train_resnext_v2.py --train_csv $TRAIN_CSV \
                       --val_csv $VAL_CSV \
                       --test_csv $TEST_CSV \
                       --batch_size 8 \
		       --use_preprocessed \
		       --alphabet $ALPHABET \
		       --epochs 200 \
                       --final_model_path $MODEL_SAVE_PATH \
                       --num_workers 4\
                       --lr 0.0003\
                       --lm_path $LM_PATH \
                       --log_dir ./log/resnet_v2_gtt_d_6_w_9_wj_4_bd_128_org_spectogram_200_test.pth \
		       --dense_dim 512 \
		       --width 9\
		       --width_jump 4\
		       --depth 6\
		       --bottleneck_depth 128\
		       --gpu_id 0\
		       --train \
		       --test \
		       --use_beamsearch \
                       --beam_width 512 \
                       --lm_alpha 0.75 \
                       --lm_beta 1 \

