#!/bin/bash
python main.py \
            --dataset wqsp \
            --model AttH \ 
            --dim 400 \
            --kg_type full \
            --valid_every 10 \ 
            --max_epochs 50 \ 
            --learning_rate 0.00002 \
            --batch_size 16 \
            --checkpoint_type libkge \ 
            --freeze True \ 
            --use_relation_matching True \
            --rel_gamma 9.5
