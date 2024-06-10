#!/bin/sh
# Sensitivity analysis for some hyper-parameters

for clip in -1 0.5 0.8 1.0 1.5 2;
  do
     echo 'Training clip:'$clip' ... ...'
     python classifier_para_SMART.py --fine-tune-mode full-model --clip_val $clip  --lr 2e-5 --gpu 1 --epochs 5
  done


for smartwei in 0 0.001 0.005 0.01 0.02 0.05 0.1 0.15 0.2 0.5;
  do
    echo 'Training smartwei:'$smartwei' ... ...'
    python classifier_para_SMART.py --fine-tune-mode full-model --smart_wei $smartwei  --lr 2e-5 --gpu 1  --epochs 5
  done

for hidden_dropout_prob in 0.01 0.05 0.1 0.2 0.3 0.5 0.6 0.9;
  do
    echo 'Training drop out :'$hidden_dropout_prob' ... ...'
    python classifier_para_SMART.py --fine-tune-mode full-model --hidden_dropout_prob $hidden_dropout_prob --clip_val 0.1 --lr 2e-5 --gpu 1  --epochs 5
  done


