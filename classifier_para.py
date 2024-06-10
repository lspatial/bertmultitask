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
import os
import pandas as pd
import multiprocessing
from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss
from transformers import get_linear_schedule_with_warmup
import gc 

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system') 

core_number = multiprocessing.cpu_count()

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

BERT_HIDDEN_SIZE = 768
TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class BertParaphraseDetector(torch.nn.Module):
    '''
    This module performs sentiment classification using BERT embeddings on the SST dataset.

    In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
    Thus, your forward() should return one logit for each of the 5 classes.
    '''

    def __init__(self, config, bert_hidden_size=BERT_HIDDEN_SIZE):
        super(BertParaphraseDetector, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'pretrain':
                param.requires_grad = False
            elif config.fine_tune_mode == 'finetune':
                param.requires_grad = True

        # Create any instance variables you need to classify the sentiment of BERT embeddings.
        ### TODO
        para_nlayers = config.sim_nlayers
        if para_nlayers is None:
            self.para_nlayers = [bert_hidden_size for i in range(config.n_hidden_layers)]
        self.dropout_paraphrase = torch.nn.ModuleList(
            [torch.nn.Dropout(config.hidden_dropout_prob) for _ in range(len(para_nlayers) + 1)])
        self.linear_paraphrase = torch.nn.ModuleList([torch.nn.Linear(bert_hidden_size, para_nlayers[0])] +
                                                     [torch.nn.Linear(para_nlayers[i - 1], para_nlayers[i]) for
                                                      i in range(1, len(para_nlayers))] + [
                                                         torch.nn.Linear(para_nlayers[-1], 1)])

    def get_sim_embeddings(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, get the BERT embeddings.'''
        # Step 0: Get [SEP] token ids
        sep_token_id = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.long, device=input_ids_1.device)
        batch_sep_token_id = sep_token_id.repeat(input_ids_1.shape[0], 1)
        input_id = torch.cat((input_ids_1, batch_sep_token_id, input_ids_2, batch_sep_token_id), dim=1)
        attention_mask = torch.cat((attention_mask_1, torch.ones_like(batch_sep_token_id), attention_mask_2,
                                    torch.ones_like(batch_sep_token_id)), dim=1)
        b_output = self.bert(input_id, attention_mask)
        cls_embeddings = b_output['pooler_output']
        return cls_embeddings #,b_output['last_hidden_state']

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        '''Takes a batch of sentences and returns logits for sentiment classes'''
        # The final BERT contextualized embedding is the hidden state of [CLS] token (the first token).
        # HINT: You should consider what is an appropriate return value given that
        # the training loop currently uses F.cross_entropy as the loss function.
        ### TODO
        crossembs = self.get_sim_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        x = crossembs
        for i in range(len(self.linear_paraphrase) - 1):
            x = self.dropout_paraphrase[i](x)
            x = self.linear_paraphrase[i](x)
            x = F.relu(x)
        x = self.dropout_paraphrase[-1](x)
        logits = self.linear_paraphrase[-1](x)
        return logits,crossembs
        # raise NotImplementedError


def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())


# Load the data: a list of (sentence, label).
def load_data(filename, flag='train'):
    data = []
    if flag == 'test':
        with open(filename, 'r', encoding="utf-8-sig") as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent_id = record['id'].lower().strip()
                data.append((preprocess_string(record['sentence1']),
                             preprocess_string(record['sentence2']),
                             sent_id))
    else:
        with open(filename, 'r', encoding="utf-8-sig") as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                try:
                    sent_id = record['id'].lower().strip()
                    data.append((preprocess_string(record['sentence1']),
                                 preprocess_string(record['sentence2']),
                                 int(float(record['is_duplicate'])), sent_id))
                except:
                    pass
        print(f"load {len(data)} data from {filename}")
    if flag == 'train':
        data=data[:len(data)] 
    return data 


# Evaluate the model on dev examples.
def model_eval(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, b_labels = (
        batch['token_ids_1'], batch['attention_mask_1'],
        batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])

        input_ids_1 = input_ids_1.to(device)
        attention_mask_1 = attention_mask_1.to(device)
        input_ids_2 = input_ids_2.to(device)
        attention_mask_2 = attention_mask_2.to(device)

        logits,_ = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        probabilities = torch.sigmoid(logits)
        threshold = 0.5
        preds = (probabilities >= threshold)
        y_true.extend(b_labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true


# Evaluate the model on test examples.
def model_test_eval(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_pred = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, sent_ids_b = (
            batch['token_ids_1'], batch['attention_mask_1'],
            batch['token_ids_2'], batch['attention_mask_2'], batch['sent_ids'])

        input_ids_1 = input_ids_1.to(device)
        attention_mask_1 = attention_mask_1.to(device)
        input_ids_2 = input_ids_2.to(device)
        attention_mask_2 = attention_mask_2.to(device)

        logits,_ = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()
        y_pred.extend(preds)
        sent_ids.extend(sent_ids_b)
    return y_pred, sent_ids


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train(args,clip_value=1,smartwei=0.02):
    if args.use_gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.use_gpu}")

    # Create the data and its corresponding datasets and dataloader.
    train_data = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')

    train_dataset = SentencePairDataset(train_data, args)
    dev_dataset = SentencePairDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn,num_workers = core_number)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn,num_workers = core_number)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'hidden_size': 768,
              'data_dir': '.',
              'sim_nlayers': args.sim_nlayers,
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = BertParaphraseDetector(config)
    model = model.to(device)

    lr = args.lr
    lr = 2e-5  # args.learning_rate - default is 5e-5, our notebook had 2e-5
    eps = 1e-8
    optimizer = AdamW(model.parameters(), lr=lr,eps=eps)
    best_dev_acc = 0
    total_steps = len(train_dataloader) * 10
    if clip_value != -1:
      print("Gradient clipped")
      for p in model.parameters():
          if p.requires_grad:
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
    else: 
      print("Gradient not clipped!!")
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    def eval_fn(embed):
        x=embed
        for i in range(len(model.linear_paraphrase) - 1):
            x = model.dropout_paraphrase[i](x)
            x = model.linear_paraphrase[i](x)
            x = F.relu(x)
        x = model.dropout_paraphrase[-1](x)
        logits = model.linear_paraphrase[-1](x)
        return logits
    # Initialize SMART regularization loss
    regularizer = SMARTLoss(
        eval_fn=eval_fn,
        loss_fn=kl_loss,  # Loss to apply between perturbed and true state
        loss_last_fn=sym_kl_loss,
        # Loss to apply between perturbed and true state on the last iteration (default = loss_fn)
    )
    # Run for the specified number of epochs.
    trainpmetrics=[]
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, b_labels = (
            batch['token_ids_1'], batch['attention_mask_1'],
            batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
            input_ids_1 = input_ids_1.to(device)
            attention_mask_1 = attention_mask_1.to(device)
            input_ids_2 = input_ids_2.to(device)
            attention_mask_2 = attention_mask_2.to(device)
            b_labels = b_labels.to(device)
            optimizer.zero_grad()
            logits,embedding = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), b_labels.float(),
                                                      reduction='sum') / len(input_ids_2)
            if args.smart:  
               loss += smartwei * regularizer(embedding, logits)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            num_batches += 1
        train_loss = train_loss / (num_batches)
        model.train(False)
        train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(
            f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        ametrics={'epoch':epoch,'loss':train_loss,'train_acc':train_acc,'dev_acc':dev_acc }
        trainpmetrics.append(ametrics)
    outfl= 'trainpmetrics'+('' if args.dataid is None else '_'+args.dataid)+'.csv'
    pd.DataFrame(trainpmetrics).to_csv(args.tarpath+'/'+outfl)
    return model

def test(args,cmodel):
    with torch.no_grad():
        if args.use_gpu < 0:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{args.use_gpu}")
        saved = torch.load(args.filepath)
        config = saved['model_config']
        model = BertParaphraseDetector(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")

        dev_data = load_data(args.dev, 'valid')
        dev_dataset = SentencePairDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=dev_dataset.collate_fn)

        test_data = load_data(args.test, 'test')
        test_dataset = SentencePairTestDataset(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=test_dataset.collate_fn)

        dev_acc, dev_f1, dev_pred, dev_true = model_eval(dev_dataloader, cmodel, device)
        print('DONE DEV')
        test_pred, sent_ids = model_test_eval(test_dataloader, model, device)
        print('DONE Test')
        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            f.write(f"id \t Paragraph_Pair \n")
            for p, s in zip(dev_true, dev_pred):
                f.write(f"{p} , {s} \n")

        with open(args.test_out, "w+") as f:
            f.write(f"id \t Paragraph_Pair \n")
            for p, s in zip(sent_ids, test_pred):
                f.write(f"{p} , {s} \n")
    testmetrics=[{'dev_acc':dev_acc }]
    outfl = 'testpmetrics' + ('' if args.dataid is None else '_' + args.dataid) + '.csv'
    pd.DataFrame(testmetrics).to_csv(args.tarpath+'/'+outfl)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--smart", action='store_true')
    parser.add_argument("--clip_val", type=float, default=1.0)
    parser.add_argument("--smart_wei", type=float, default=0.02) 
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=32) ## 12 good!!!
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-3)
    parser.add_argument("--root", type=str, help="root path for training",
                        default='/adt_geocomplex/bertresult_para_testing')
    args = parser.parse_args()
    return args

def retrievePath(args,tduty,other=None):
    tpath=(args.root+'/'+tduty+'.ly'+'_'.join(str(x) for x in args.sim_nlayers)+'.hz'+ str(args.hidden_size)+
           '.'+args.fine_tune_mode+'.dp'+str(args.hidden_dropout_prob)+'.lr'+str(args.lr)+('_smart' if args.smart else ''))
    if other is not None:
        tpath += '.'+other
    if os.path.exists(tpath):
        subfolders = [ os.path.basename(f) for f in os.scandir(tpath) if f.is_dir() and os.path.basename(f).isdigit()]
        if len(subfolders) > 0:
            res = [eval(i) for i in subfolders]
            maxres = max(res)
            tpath +=  '/' + str(maxres+1)
        else:
            tpath +=  '/' + str(1)
    else:
        tpath += '/1'
    os.makedirs(tpath, exist_ok=True)
    os.makedirs(tpath + '/predictions/', exist_ok=True)
    return tpath

def getconfig(args,tarpath):
   config = SimpleNamespace(
        filepath=tarpath + '/quora_classifier.pt',
        dataid=None,
        lr=args.lr,
        sim_nlayers=args.sim_nlayers,
        data_dir='.',
        hidden_size=args.hidden_size,
        use_gpu=args.gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='data/quora-train.csv',
        dev='data/quora-dev.csv',
        test='data/quora-test-student.csv',
        fine_tune_mode=args.fine_tune_mode,
        tarpath=tarpath,
        smart=args.smart,
        dev_out=tarpath + '/predictions/' + args.fine_tune_mode + '-quora-dev-out.csv',
        test_out=tarpath + '/predictions/' + args.fine_tune_mode + '-quora-test-out.csv'
    )
   return config 


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    args.sim_nlayers = [768]
    clip_value = float(args.clip_val)
    smart_wei = float(args.smart_wei)    
    dropout= float(args.hidden_dropout_prob)
    tarpath = retrievePath(args, tduty='para_clip_'+str(clip_value)+'_swei_'+ str(smart_wei) + '_dropout_'+str(dropout), other=None)
    config = getconfig(args,tarpath)
    model=train(config,clip_value=clip_value,smartwei=smart_wei)
    del model 
    torch.cuda.empty_cache()
    gc.collect()  
    print('Evaluating on QUORA...')
    #test(config,model)
