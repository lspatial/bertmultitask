#!/bin/sh

#python multitask_classifier.py --fine-tune-mode last-linear-layer  --lr 1e-4 --use_gpu 1 --gradient_surgery  --mark_regularizer --use_amp   --batch_size 64

# python multitask_classifier.py --fine-tune-mode full-model --lr 1e-5 --use_gpu 1 --gradient_surgery  --mark_regularizer --use_amp   --batch_size 64  
 
#python multitask_classifier.py --fine-tune-mode full-model --lr 2e-5 --use_gpu 1 --mark_regularizer  --batch_size 64  
 
#python multitask_classifier.py  --fine-tune-mode full-model --lr 2e-5 --use_gpu 0 --gradient_surgery  --mark_regularizer --use_amp   --batch_size 32  --mul_bentropy  --data_aug 

python multitask_classifier.py  --fine-tune-mode full-model --lr 1e-5 --use_gpu 0  --batch_size 32  
