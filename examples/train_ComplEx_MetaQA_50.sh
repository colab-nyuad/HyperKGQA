#!/bin/bash
python main.py \
            --dataset MetaQA \
            --model ComplEx \ 
            --dim 50 \
            --kg_type half \
            --valid_every 10 \
            --max_epochs 50 \
            --learning_rate 0.0005 \
            --checkpoint_type ldh \
            --hops 1 \
