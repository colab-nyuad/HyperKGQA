#!/bin/bash
python main.py \
            --dataset MetaQA \
            --model AttH \ 
            --dim 50 \
            --kg_type full \
            --valid_every 10 \
            --max_epochs 50 \
            --learning_rate 0.0005 \
            --checkpoint_type ldh \
            --hops 3
