import os.path

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
import sys
import gzip
from datetime import datetime

import random, numpy as np, argparse
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

def retrievetexts(files,modes):
    train_sentences = []
    for i,f in enumerate(files):
        with open(f, 'r', encoding='utf-8-sig') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                if modes[i]=='senti':
                    train_sentences.append(record['sentence'].lower().strip())
                elif modes[i]=='para' or modes[i]=='sim':
                    train_sentences.append(record['sentence1'].lower().strip())
                    train_sentences.append(record['sentence2'].lower().strip())
    return train_sentences

# A dataset wrapper, that tokenizes our data on-the-fly
class TokenizedSentencesDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        if not self.cache_tokenization:
            return self.tokenizer(
                self.sentences[item],
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
            )

        if isinstance(self.sentences[item], str):
            self.sentences[item] = self.tokenizer(
                self.sentences[item],
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
            )
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)

def retrieveConfig():
    config={}
    config['per_device_train_batch_size'] = 16
    config['save_steps'] = 1000  # Save model every 1k steps
    config['num_train_epochs'] = 3  # Number of epochs
    config['use_fp16'] = False  # Set to True, if your GPU supports FP16 operations
    config['max_length'] = 100  # Max length for a text input
    config['do_whole_word_mask'] = True  # If set to true, whole words are masked
    config['mlm_prob'] = 0.15  # Probability that a word is replaced by a [MASK] token
    return config

def pretrainBERT2CustomData(model_name,pconfig,files, modes,output_dir):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_sentences = retrievetexts(files, modes)
    dev_dataset = None # TFAutoModelForPreTraining 
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    #model = BertModel.from_pretrained(model_name)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer,tokenizer.model_max_length)
    if pconfig['do_whole_word_mask']:
        data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=pconfig['mlm_prob'])
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=pconfig['mlm_prob'])
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=pconfig['num_train_epochs'],
        evaluation_strategy="steps" if dev_dataset is not None else "no",
        per_device_train_batch_size=pconfig['per_device_train_batch_size'],
        eval_steps=pconfig['save_steps'],
        save_steps=pconfig['save_steps'],
        logging_steps=pconfig['save_steps'],
        save_total_limit=1,
        prediction_loss_only=True,
        fp16=pconfig['use_fp16'],
    )
    trainer = Trainer(
        model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset, eval_dataset=dev_dataset
    )
    print("Save tokenizer to:", output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.train()
    print("Save model to:", output_dir)
    model.save_pretrained(output_dir)
    print("Training done")

def main():
    model_name = 'bert-base-uncased'
    pconfig = retrieveConfig()
    databpath = './data'
    files = ['ids-sst-train.csv','ids-sst-dev.csv','ids-cfimdb-train.csv','ids-cfimdb-dev.csv',
             'quora-train.csv','quora-dev.csv','sts-train.csv','sts-dev.csv' ]
    files =[databpath + '/' + f for f in files ]
    modes = ['senti','senti','senti','senti','para','para','sim','sim']
    output_dir = './pretrainedBert'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    pretrainBERT2CustomData(model_name, pconfig, files, modes, output_dir)

if __name__ == "__main__":
    main()

