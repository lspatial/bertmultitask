# `Enhancing BERT's Performance on Downstream Tasks via Multitask Fine-Tuning and Ensembling `

This is the default final project for the Stanford CS 224N class.

Lianfa Li  Email: lianfali@stanford.edu



## Abstract 

Fine-tuning pretrained language models like BERT on diverse downstream tasks via multitask learning presents challenges stemming from task discrepancies that can impede performance. This project proposes a multifaceted approach to enhance BERT's multitask capabilities through architectural modifications, regularization techniques, and optimized training methods. Our exploration uncovered that top layer adaptations, gradient clipping, and SMART regularization significantly boosted individual task performance. For multitask models, shared projected attention layers, round-robin sampling, and strategic gradient surgery emerged as pivotal factors. Ensemble strategies incorporating voting mechanisms further improved the total accuracy to 0.796 on the test leaderboard. Our study underscores the significance of judiciously incorporating shared and task-specific network modifications, gradient control methods, regularization, and sampling strategies to optimize the transferability of large language models across diverse tasks.

## Net structures for tesing

![netstructures.png](assets/netstructures.png)

## Test results


| Overall Test Score | SST Test Acc | Para QQR Test Acc | STS Test Cor |  |
| ------------------ | ------------ | :----------------: | ------------ | - |
| 0.796              | 0.541        |       0.901       | 0.891        |  |

## Commands

Pease run the following command for individual task model:

(1) Run the model for sentiment analysis

`python classifier_senti.py --fine-tune-mode full-model --lr 1e-5 --gpu 0  --epochs 8`

(2) Run the model for paraphrase detection

`python classifier_para_SMART.py --fine-tune-mode full-model --lr 2e-5 --gpu 1 --smart6`

(3) Run the model for similarity estimation of pair sentences

`python classifier_sim_SMART.py --fine-tune-mode full-model --lr 1e-5 --gpu 0`

optimizer.py: Missing code blocks.

2. Please run the following command for multitask modeling:

(1) Run the mutitask model with SMART regularizer, gradient surgery and robin round sampling

`python multitask_classifier.py  --fine-tune-mode full-model --lr 2e-5 --use_gpu 0  --gradient_surgery  --mark_regularizer --use_amp   --batch_size 32  --mul_bentropy  --data_aug`

(2) Run the mutitask model of shared and specific-tast PALs with SMART regularizer, gradient surgery and robin round sampling

`python multitask_classifier_cpal2.py --fine-tune-mode full-model --lr 1e-5 --use_gpu 0 --gradient_p surgery  --mark_regularizer --sentiloss both --use_amp  --batch_size 32  --data_aug --epochs 6`

(3) Run the mutitask model of shared Adaptors with SMART regularizer, gradient surgery and robin round sampling

`python multitask_classifier_adaptor.py --fine-tune-mode full-model --lr 1e-5 --use_gpu 0   --gradient_p surgery  --mark_regularizer   --mul_bentropy --use_amp  --batch_size 32  --data_aug --epochs 8`

(4) Run the mutitask model of saparate task modules (not shared parameters between tasks)

`python multitask_classifier_sepbert2_early.py --fine-tune-mode full-model --lr 1e-5 --use_gpu 0  --mark_regularizer  --use_amp  --batch_size 32  --data_aug --epochs 30`

For sensitivity analysis, please refer the code of the code file, run_sensitivity_ana.sh.

3. Please run the following command for ensemble predictions:

`python ensemblebert.py `

## Setup instructions

Follow `setup.sh` to properly setup a conda environment and install dependencies.

## Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
