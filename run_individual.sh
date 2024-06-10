#!/bin/sh

#Run the individual task model, please use --help to see the detailed explainations for each argument
#Run the model for sentiment analysis
python classifier_senti.py --fine-tune-mode full-model --lr 1e-5 --gpu 0  --epochs 8

#Run the model for paraphrase detection
python classifier_para_SMART.py --fine-tune-mode full-model --lr 2e-5 --gpu 1 --smart

#Run the model for similarity estimation of pair sentences
python classifier_sim_SMART.py --fine-tune-mode full-model --lr 1e-5 --gpu 0






