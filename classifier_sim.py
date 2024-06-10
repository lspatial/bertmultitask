import os.path
import random, numpy as np, argparse
from types import SimpleNamespace
import csv

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm
from glob import glob
from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss
from transformers import get_linear_schedule_with_warmup


from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

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


class BertSimilarity(torch.nn.Module):
    '''
    This module performs sentiment classification using BERT embeddings on the SST dataset.

    In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
    Thus, your forward() should return one logit for each of the 5 classes.
    '''

    def __init__(self, config, bert_hidden_size=BERT_HIDDEN_SIZE):
        super(BertSimilarity, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True

        # Create any instance variables you need to classify the sentiment of BERT embeddings.
        ### TODO
        sim_nlayers = config.sim_nlayers
        self.dropout_similarity = torch.nn.ModuleList(
            [torch.nn.Dropout(config.hidden_dropout_prob) for _ in range(len(sim_nlayers) + 1)])
        self.linear_similarity = torch.nn.ModuleList([torch.nn.Linear(bert_hidden_size, sim_nlayers[0])] +
                                              [torch.nn.Linear(sim_nlayers[i - 1], sim_nlayers[i]) for i in
                                               range(1, len(sim_nlayers))] + [
                                                  torch.nn.Linear(sim_nlayers[-1], 1)])
        # raise NotImplementedError

    def get_sim_embeddings(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, get the BERT embeddings.'''
        # Step 0: Get [SEP] token ids
        sep_token_id = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.long, device=input_ids_1.device)
        batch_sep_token_id = sep_token_id.repeat(input_ids_1.shape[0], 1)
        input_id = torch.cat((input_ids_1, batch_sep_token_id, input_ids_2, batch_sep_token_id), dim=1)
        attention_mask = torch.cat((attention_mask_1, torch.ones_like(batch_sep_token_id), attention_mask_2,
                                    torch.ones_like(batch_sep_token_id)), dim=1)
        b_output = self.forward(input_id, attention_mask)
        return b_output

    def forward(self, input_ids, attention_mask):
        '''Takes a batch of sentences and returns logits for sentiment classes'''
        # The final BERT contextualized embedding is the hidden state of [CLS] token (the first token).
        # HINT: You should consider what is an appropriate return value given that
        # the training loop currently uses F.cross_entropy as the loss function.
        ### TODO
        bert_output = self.bert(input_ids, attention_mask)
        cls_embeddings = bert_output['pooler_output']
        return cls_embeddings
        # raise NotImplementedError

    def tasklayer_similarity(self,x):
        for i in range(len(self.linear_similarity) - 1):
            x = self.dropout_similarity[i](x)
            x = self.linear_similarity[i](x)
            x = F.relu(x)
        x = self.dropout_similarity[-1](x)
        y = self.linear_similarity[-1](x)
        return y

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        embds = self.get_sim_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        y = self.tasklayer_similarity(embds)
        return y



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
                                 int(float(record['similarity'])), sent_id))
                except:
                    pass
        print(f"load {len(data)} data from {filename}")
    return  data 


# Evaluate the model on dev examples.
def model_eval(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_true = []
    y_pred = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, y_true_b = (
        batch['token_ids_1'], batch['attention_mask_1'],
        batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])

        input_ids_1 = input_ids_1.to(device)
        attention_mask_1 = attention_mask_1.to(device)
        input_ids_2 = input_ids_2.to(device)
        attention_mask_2 = attention_mask_2.to(device)
        preds = model.predict_similarity(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        preds = preds.flatten()
        preds = preds.detach().cpu().numpy()
        y_true.extend(y_true_b)
        y_pred.extend(preds)
    pearson_mat = np.corrcoef(y_pred, y_true)
    sts_corr = pearson_mat[1][0]
    return sts_corr, y_pred, y_true, sent_ids


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
        preds = model.predict_similarity(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
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


def train(args):
    if args.use_gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.use_gpu}")
    # Create the data and its corresponding datasets and dataloader.
    train_data = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')

    train_dataset = SentencePairDataset(train_data, args, isRegression=True)
    dev_dataset = SentencePairDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'hidden_size': 768,
              'data_dir': '.',
              'sim_nlayers':args.sim_nlayers,
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)
    model = BertSimilarity(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_cor = 0

    clip_value = 0.5
    print("Gradient clipped")
    for p in model.parameters():
      if p.requires_grad:
          p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
    best_dev_acc = 0
    total_steps = len(train_dataloader) * 10
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    smart_regularizer = SMARTLoss(eval_fn=model.tasklayer_similarity, loss_fn=F.mse_loss, loss_last_fn=F.mse_loss,
                                  num_steps=1,  # Number of optimization steps to find noise (default = 1)
                                  step_size=1e-5,  # Step size to improve noise (default = 1e-3)
                                  epsilon=1e-6,  # Noise norm constraint (default = 1e-6)
                                  noise_var=1e-6  # Initial noise variance (default = 1e-5)
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
            embeddings = model.get_sim_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            preds = model.tasklayer_similarity(embeddings)
            loss = F.mse_loss(preds.view(-1), b_labels.view(-1), reduction='sum') / args.batch_size
            loss += 0.2*smart_regularizer(embeddings, preds)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            num_batches += 1
        train_loss = train_loss / (num_batches)

        train_cor, *_ = model_eval(train_dataloader, model, device)
        dev_cor, *_ = model_eval(dev_dataloader, model, device)

        if dev_cor > best_dev_cor:
            best_dev_cor = dev_cor
            save_model(model, optimizer, args, config, args.filepath)
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train cor:: {train_cor :.3f}, dev cor:: {dev_cor:.3f}")
        ametrics={'epoch':epoch,'loss':train_loss,'train_cor':train_cor,'dev_cor':dev_cor }
        trainpmetrics.append(ametrics)
    outfl= 'trainpmetrics'+('' if args.dataid is None else '_'+args.dataid)+'.csv'
    pd.DataFrame(trainpmetrics).to_csv(args.tarpath+'/'+outfl)

def test(args):
    with torch.no_grad():
        if args.use_gpu < 0:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{args.use_gpu}")
        saved = torch.load(args.filepath)
        config = saved['model_config']
        model = BertSimilarity(config)
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
        dev_cor, dev_pred, dev_true, dev_sent_ids = model_eval(dev_dataloader, model, device)
        print('DONE DEV')
        test_pred, test_sent_ids = model_test_eval(test_dataloader, model, device)
        print('DONE Test')
        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_cor :.3f}")
            f.write(f"id \t Predicted_Similarity \n")
            for p, s in zip(dev_sent_ids, dev_pred):
                f.write(f"{p} , {s} \n")

        with open(args.test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sent_ids, test_pred):
                f.write(f"{p} , {s} \n")
    testmetrics=[{'dev_cor':dev_cor }]
    outfl = 'testpmetrics' + ('' if args.dataid is None else '_' + args.dataid) + '.csv'
    pd.DataFrame(testmetrics).to_csv(args.tarpath+'/'+outfl)

def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=48)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-3)
    parser.add_argument("--root", type=str, help="root path for training",
                        default='/adt_geocomplex/bertresult_simtest')

    args = parser.parse_args()
    return args

def retrievePath(args,tduty,other=None):
    tpath=(args.root+'/'+tduty+'.ly'+'_'.join(str(x) for x in args.sim_nlayers)+'.hz'+ str(args.hidden_size)+
           '.'+args.fine_tune_mode+'.dp'+str(args.hidden_dropout_prob)+'.lr'+str(args.lr))
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
    print('Training Sentence Similarity on STS...')
    args.sim_nlayers = [768]
    tarpath=retrievePath(args, tduty='sim', other=None)
    config = SimpleNamespace(
        filepath=tarpath+'/sentencesim.pt',
        dataid=None,
        lr=args.lr,
        sim_nlayers=args.sim_nlayers,
        data_dir='.',
        num_labels=1,
        hidden_size=args.hidden_size,
        use_gpu=args.gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='data/sts-train.csv',
        dev='data/sts-dev.csv',
        test='data/sts-test-student.csv',
        fine_tune_mode=args.fine_tune_mode,
        tarpath=tarpath,
        dev_out=tarpath+'/predictions/' + args.fine_tune_mode + '-sts-dev-out.csv',
        test_out=tarpath+'/predictions/' + args.fine_tune_mode + '-sts-test-out.csv'
    )

    train(config)

    print('Evaluating on STS...')
    test(config)
