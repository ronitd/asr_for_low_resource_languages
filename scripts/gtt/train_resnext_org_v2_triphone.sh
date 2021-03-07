#!/bin/bash
#TRAIN_CSV="$HOME/Speech/mix-gtt-train-time.csv"
#TRAIN_CSV="$HOME/Speech/Tamil/test.csv"
TRAIN_CSV="$HOME/Speech/Gujarati/gu-train.csv"
TRAIN_CSV="$HOME/Speech/Gujarati/gu-train-time-short.csv"
TEST_CSV="$HOME/Speech/mix-gtt-test.csv"
TEST_CSV="$HOME/Speech/Gujarati/gu-test-100.csv"
TIMESTEP=12
BATCH_SIZE=16
ALPHABETFILE=/shared/kgcoe-research/mil/SpeechCorpora/LibriSpeech_preprocess/english_alphabet.txt
TAMIL_ALPHABET="$HOME/Speech/Tamil/ta-alphabet.txt"
TAMIL_ALPHABET="$HOME/Speech/Gujarati/gu-alphabet.txt"
PHONEME_DICT="$HOME/Speech/labels.pkl"
LEXICON="$HOME/Speech/lexicon.pkl"
PHONEME_WORD="$HOME/Speech/Gujarati/gu-phoneme-word.pkl" 
TRIE="$HOME/Speech/Gujarati/gu-trie.pkl"

TRIPHONE_TO_INT="$HOME/Speech/Gujarati/gu-triphones-to-num-labels.pkl"
INT_TO_TRIPHONE="$HOME/Speech/Gujarati/gu-num-labels-to-triphones.pkl"
WORD_TO_TRIPHONES_LABEL="$HOME/Speech/Gujarati/gu-word-to-triphone-label-int.pkl"

MODEL_SAVE_PATH=../../train_models/resnet_v2_triphone_gtt_d_4_w_4_wj_4_bd_16_org_spectogram.pth

LM_PATH="$HOME/Speech/Tamil/ta-lm.binary"
LM_PATH="$HOME/Speech/Gujarati/gu-srilm-kd-3-lm-phoneme.arpa"
PREVIOUS_MODEL=../../train_models/resnet_v2_phoneme_gtt_d_6_w_9_wj_4_bd_128_org_spectogram.pth
PREVIOUS_MODEL=../../train_models/resnet_v2_phoneme_gtt_d_6_w_9_wj_4_bd_128_org_spectogram_200_300.pth
#Original lr: 0.0003 -- tf: 0.00005
#export CUDA_VISIBLE_DEVICES=0
python3 -W ignore ../../train_resnext_v2_triphone.py --train_csv $TRAIN_CSV \
                       --val_csv $TEST_CSV \
                       --test_csv $TEST_CSV \
                       --triphone_to_int $TRIPHONE_TO_INT \
                       --int_to_triphone $INT_TO_TRIPHONE \
                       --word_to_triphone_label $WORD_TO_TRIPHONES_LABEL \
                       --batch_size 4 \
                       --labels $PHONEME_DICT \
                       --use_preprocessed \
                       --lexicon $LEXICON \
                       --alphabet $TAMIL_ALPHABET \
                       --phoneme_vocab $PHONEME_WORD \
                       --trie $TRIE \
                       --epochs 100 \
                       --final_model_path $MODEL_SAVE_PATH \
                       --num_workers 4\
                       --lr 0.0003\
                       --lm_path $LM_PATH \
                       --log_dir ./log/resnet_v2_triphone_gtt_d_4_w_4_wj_4_bd_16_org_spectogram.pth \
                       --dense_dim 256 \
                       --width 4\
                       --width_jump 4\
                       --depth 4\
                       --bottleneck_depth 16\
                       --gpu_id 0\
		       --train \
                       #--continue_from $PREVIOUS_MODEL\
                       #--test \
                       #--beam_width 64 \
                       #--lm_alpha 0.5 \
                       #--lm_beta 0.5 \
		               #--use_beamsearch \

