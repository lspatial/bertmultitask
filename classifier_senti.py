import random, numpy as np, argparse
from types import SimpleNamespace
import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

import pandas as pd
from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss
from transformers import get_linear_schedule_with_warmup

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

TQDM_DISABLE=False
BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5 

# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class BertSentimentClassifier(torch.nn.Module):
    '''
    This module performs sentiment classification using BERT embeddings on the SST dataset.

    In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
    Thus, your forward() should return one logit for each of the 5 classes.
    '''
    def __init__(self, config, bert_hidden_size=BERT_HIDDEN_SIZE):
        super(BertSentimentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Pretrain mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True

        # Create any instance variables you need to classify the sentiment of BERT embeddings.
        ### TODO
        self.dropout_sentiment = nn.ModuleList(
            [nn.Dropout(config.hidden_dropout_prob) for _ in range(config.n_hidden_layers + 1)])
        self.linear_sentiment = nn.ModuleList(
            [nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE) for _ in range(config.n_hidden_layers)] + [
                nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)])
                
        if config.sentiloss == 'both':
          self.dropout_sentiment2 = nn.ModuleList(
              [nn.Dropout(config.hidden_dropout_prob) for _ in range(config.n_hidden_layers + 1)])
          self.linear_sentiment2 = nn.ModuleList(
              [nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE) for _ in range(config.n_hidden_layers)] + [
                  nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)])  
        #raise NotImplementedError


    def forward(self, input_ids, attention_mask):
        '''Takes a batch of sentences and returns logits for sentiment classes'''
        # The final BERT contextualized embedding is the hidden state of [CLS] token (the first token).
        # HINT: You should consider what is an appropriate return value given that
        # the training loop currently uses F.cross_entropy as the loss function.
        ### TODO
        bert_output = self.bert(input_ids, attention_mask)
        pool_output = bert_output['pooler_output']
        return pool_output
      
      
    def tasklayer_sentiment(self,x):
        for i in range(len(self.linear_sentiment) - 1):
            x = self.linear_sentiment[i](x)
            x = self.linear_sentiment[i](x)
            x = F.relu(x)
        x = self.dropout_sentiment[-1](x)
        logits = self.linear_sentiment[-1](x)
        return logits
      
    def tasklayer_sentiment2(self,x):
        for i in range(len(self.linear_sentiment2) - 1):
            x = self.linear_sentiment2[i](x)
            x = self.linear_sentiment2[i](x)
            x = F.relu(x)
        x = self.dropout_sentiment2[-1](x)
        logits = self.linear_sentiment2[-1](x)
        return logits    
       # raise NotImplementedError
       
       
    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        embeds = self.forward(input_ids, attention_mask)
        if args.sentiloss == 'both':
           y = self.tasklayer_sentiment(embeds)
           y2 = self.tasklayer_sentiment2(embeds)
           return y,y2
        y = self.tasklayer_sentiment(embeds)
        return y    
       

class SentimentDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


class SentimentTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


# Load the data: a list of (sentence, label).
def load_data(filename, flag='train'):
    num_labels = {}
    data = []
    if flag == 'test':
        with open(filename, 'r',encoding='utf-8-sig') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                data.append((sent,sent_id))
    else:
        with open(filename, 'r',encoding='utf-8-sig') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                data.append((sent, label,sent_id))
        print(f"load {len(data)} data from {filename}")
#    data = data[:500]
    if flag == 'train':
        return data, len(num_labels)
    else:
        return data


# Evaluate the model on dev examples.
def model_eval(dataloader, model, device,sentiloss='both'):
    model.eval() # Switch to eval model, will turn off randomness like dropout.
    y_true = []
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'],batch['attention_mask'],  \
                                                        batch['labels'], batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        if sentiloss == 'both':
                logits, logits2 = model.predict_sentiment(b_ids, b_mask)
                logits = logits.detach().cpu().numpy()
                logits2 = logits2.detach().cpu().numpy()
                y_hat = np.argmax(logits+logits2, axis=1).flatten()
        else:
                logits = model.predict_sentiment(b_ids, b_mask)
                logits = logits.detach().cpu().numpy()
                y_hat = np.argmax(logits, axis=1).flatten()
        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(y_hat)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents, sent_ids


# Evaluate the model on test examples.
def model_test_eval(dataloader, model, device,sentiloss=None):
    model.eval() # Switch to eval model, will turn off randomness like dropout.
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_sents, b_sent_ids = batch['token_ids'],batch['attention_mask'],  \
                                                         batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        if sentiloss == 'both':
                logits, logits2 = model.predict_sentiment(b_ids, b_mask)
                logits = logits.detach().cpu().numpy()
                logits2 = logits2.detach().cpu().numpy()
                y_hat = np.argmax(logits+logits2, axis=1).flatten()
        else:
                logits = model.predict_sentiment(b_ids, b_mask)
                logits = logits.detach().cpu().numpy()
                y_hat = np.argmax(logits, axis=1).flatten()
        y_pred.extend(y_hat)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    return y_pred, sents, sent_ids


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


def train(args):
    if args.use_gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.use_gpu}")
    # Create the data and its corresponding datasets and dataloader.
    train_data, num_labels = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')

    train_dataset = SentimentDataset(train_data, args)
    dev_dataset = SentimentDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'sentiloss':args.sentiloss,
              'n_hidden_layers':args.n_hidden_layers,
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = BertSentimentClassifier(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0
    clip_value=0.1
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
    # Run for the specified number of epochs.
    # Initialize SMART regularization loss
    regularizer = SMARTLoss(
        eval_fn=model.tasklayer_sentiment,
        loss_fn=kl_loss,  # Loss to apply between perturbed and true state
        loss_last_fn=sym_kl_loss,
        # Loss to apply between perturbed and true state on the last iteration (default = loss_fn)
    )
    smartwei=0.01
    trainpmetrics=[]
    criteria_mbce=torch.nn.BCEWithLogitsLoss()
    total_steps = len(train_dataloader) * 8
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            embeddings = model.forward(b_ids, b_mask)
            logits = model.tasklayer_sentiment(embeddings)
            if args.sentiloss == 'mul_bentropy':
                multi_labels = torch.zeros(len(logits), 5).to(device)
                # Scatter 1s in the corresponding positions of single_labels
                multi_labels[range(len(logits)), b_labels.view(-1)] = 1
                loss = F.binary_cross_entropy_with_logits(logits, multi_labels)
            elif args.sentiloss == 'both':
                logits2 = model.tasklayer_sentiment2(embeddings)
                multi_labels = torch.zeros(len(logits), 5).to(device)
                # Scatter 1s in the corresponding positions of single_labels
                multi_labels[range(len(logits)), b_labels.view(-1)] = 1
                loss1 = F.binary_cross_entropy_with_logits(logits, multi_labels)
                loss2 = F.cross_entropy(logits2, b_labels.view(-1), reduction='mean')
                loss = loss1 + loss2 
            else:
                loss = F.cross_entropy(logits, b_labels.view(-1))
            loss += smartwei * regularizer(embeddings, logits)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_  = model_eval(train_dataloader, model, device,args.sentiloss)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device,args.sentiloss)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        ametrics={'epoch':epoch,'loss':train_loss,'train_acc':train_acc,'dev_acc':dev_acc }
        trainpmetrics.append(ametrics)
    outfl= 'trainpmetrics'+('' if args.dataid is None else '_'+args.dataid)+'.csv'
    pd.DataFrame(trainpmetrics).to_csv(args.tarpath+'/'+outfl)

def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']
        model = BertSentimentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")
        
        dev_data = load_data(args.dev, 'valid')
        dev_dataset = SentimentDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

        test_data = load_data(args.test, 'test')
        test_dataset = SentimentTestDataset(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
        
        dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_dataloader, model, device,args.sentiloss)
        print('DONE DEV')
        test_pred, test_sents, test_sent_ids = model_test_eval(test_dataloader, model, device,args.sentiloss)
        print('DONE Test')
        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sent_ids,dev_pred ):
                f.write(f"{p} , {s} \n")

        with open(args.test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s  in zip(test_sent_ids,test_pred ):
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
    parser.add_argument("--gpu", type=int, default=0)                     
    parser.add_argument("--sentiloss", type=str,
                        help='mul_bentropy: multiple binary cross entropy; None (default): cross entropy; both: entropy and multipe binary cross entropy',
                        choices=('mul_bentropy', 'both'), default=None)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--n_hidden_layers", type=int, default=2, help="Number of hidden layers for the classifier")
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-3)
    parser.add_argument("--root", type=str, help="root path for training",
                        default='/adt_geocomplex/bertresult/sentiness_default')
    args = parser.parse_args()
    return args

def retrievePath(args,tduty,other=None):
    tpath=(args.root+'/'+tduty+'.ly'+'_'.join(str(x) for x in args.sim_nlayers)+'.hz'+ str(args.hidden_size)+
           '.'+args.fine_tune_mode+'.dp'+str(args.hidden_dropout_prob)+'.lr'+str(args.lr))+'_smart'
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


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    print('Training Sentiment Classifier on SST...')
    args.sim_nlayers =[768, 512]
    tarpath = retrievePath(args, tduty='senti', other=None)

    config = SimpleNamespace(
        filepath=tarpath + '/sst_classifier.pt',
        dataid = 'sst',
        num_labels = 5,
        lr=args.lr,
        sim_nlayers=args.sim_nlayers,
        data_dir='.',
        hidden_size=args.hidden_size,
        use_gpu=args.gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_hidden_layers=args.n_hidden_layers, 
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='data/ids-sst-train.csv',
        dev='data/ids-sst-dev.csv',
        test='data/ids-sst-test-student.csv',
        fine_tune_mode=args.fine_tune_mode,
        tarpath=tarpath,
        sentiloss=args.sentiloss,
        dev_out=tarpath + '/predictions/' + args.fine_tune_mode + '-sst-dev-out.csv',
        test_out=tarpath + '/predictions/' + args.fine_tune_mode + '-sst-test-out.csv'
    )

    train(config)

    print('Evaluating on SST...')
    test(config)

    config = SimpleNamespace(
        filepath=tarpath + '/cfimdb_classifier.pt',
        dataid='fimdb',
        num_labels = 5,
        lr=args.lr,
        sim_nlayers=args.sim_nlayers,
        data_dir='.',
        hidden_size=args.hidden_size,
        use_gpu=args.gpu,
        epochs=args.epochs,
        n_hidden_layers=args.n_hidden_layers, 
        batch_size=args.batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='data/ids-cfimdb-train.csv',
        dev='data/ids-cfimdb-dev.csv',
        test='data/ids-cfimdb-test-student.csv',
        fine_tune_mode=args.fine_tune_mode,
        tarpath=tarpath,
        sentiloss=args.sentiloss,
        dev_out=tarpath + '/predictions/' + args.fine_tune_mode + '-cfimdb-dev-out.csv',
        test_out=tarpath + '/predictions/' + args.fine_tune_mode + '-cfimdb-test-out.csv'
    )

    #train(config)

    #print('Evaluating on cfimdb...')
    #test(config)
