#!/bin/sh

#Run the multitask model for sentiment analysis, paraphrase detection and similarity, please use --help to see the detailed explainations for each argument

#Run the mutitask model with SMART regularizer, gradient surgery and robin round sampling
python multitask_classifier.py  --fine-tune-mode full-model --lr 2e-5 --use_gpu 0 \
       --gradient_surgery  --mark_regularizer --use_amp   --batch_size 32  --mul_bentropy  --data_aug

#Run the mutitask model of shared and specific-tast PALs with SMART regularizer, gradient surgery and robin round sampling
python multitask_classifier_cpal2.py --fine-tune-mode full-model --lr 1e-5 --use_gpu 0 \
              --gradient_p surgery  --mark_regularizer --sentiloss both --use_amp \
              --batch_size 32  --data_aug --epochs 6

#Run the mutitask model of shared Adaptors with SMART regularizer, gradient surgery and robin round sampling
python multitask_classifier_adaptor.py --fine-tune-mode full-model --lr 1e-5 --use_gpu 0 \
              --gradient_p surgery  --mark_regularizer   --mul_bentropy --use_amp \
              --batch_size 32  --data_aug --epochs 8

#Run the mutitask model of saparate task modules (not shared parameters between tasks)
python multitask_classifier_sepbert2_early.py --fine-tune-mode full-model --lr 1e-5 --use_gpu 0 \
              --mark_regularizer  --use_amp \
              --batch_size 32  --data_aug --epochs 30








