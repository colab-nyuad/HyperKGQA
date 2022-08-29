#!/bin/bash
python main.py \
            --dataset wqsp \
            --model AttH \ 
            --dim 50 \
            --kg_type half \
            --valid_every 10 \ 
            --max_epochs 50 \ 
            --learning_rate 0.00002 \
            --batch_size 16 \
            --checkpoint_type libkge \ 
            --freeze True \ 
