#!/bin/sh

python multitask_classifier_adaptor.py --fine-tune-mode full-model --lr 1e-5 --use_gpu 0 \
              --gradient_p surgery  --mark_regularizer   --mul_bentropy --use_amp \
              --batch_size 32  --data_aug --epochs 8 


#python multitask_classifier_adaptor.py --fine-tune-mode last-linear-layer --lr 2e-3 --use_gpu 0 \
##              --gradient_surgery  --mark_regularizer  --mul_bentropy --use_amp \
#              --batch_size 32  --data_aug --epochs 8  


#History 
#python multitask_classifier_adaptor.py --fine-tune-mode full-model --lr 2e-5 --use_gpu 0 \
#              --gradient_surgery  --mark_regularizer  --mul_bentropy --use_amp \
#              --batch_size 32  --data_aug --epochs 6  


#python multitask_classifier_adaptor.py --fine-tune-mode full-model --lr 3e-5 --use_gpu 0 \
#              --gradient_surgery  --mark_regularizer  --test_mode --mul_bentropy --use_amp \
#              --batch_size 32  --data_aug --epochs 1  
 

