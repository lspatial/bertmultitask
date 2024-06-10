'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm
import pandas as pd
from torch.cuda.amp import GradScaler, autocast
from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss
from transformers import get_linear_schedule_with_warmup
from helper import smart_regularizer, retrievePathold
from itertools import cycle
from torch.cuda.amp import GradScaler, autocast
from pcgrad import PCGrad
import gc 

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask

TQDM_DISABLE=False

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        #Goal 1: Add a linear layer for sentiment classification
        self.dropout_sentiment = nn.ModuleList(
            [nn.Dropout(config.hidden_dropout_prob) for _ in range(config.n_hidden_layers + 1)])
        self.linear_sentiment = nn.ModuleList(
            [nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE) for _ in range(config.n_hidden_layers)] + [
                nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)])

        #Goal 2: Add a linear layer for paraphrase detection
        self.dropout_paraphrase = nn.ModuleList(
            [nn.Dropout(config.hidden_dropout_prob) for _ in range(config.n_hidden_layers + 1)])
        self.linear_paraphrase = nn.ModuleList(
            [nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE) for _ in range(config.n_hidden_layers)] + [
                nn.Linear(BERT_HIDDEN_SIZE, 1)])

        # Goal 3: Add a linear layer for semantic textual similarity
        self.dropout_similarity = nn.ModuleList(
            [nn.Dropout(config.hidden_dropout_prob) for _ in range(config.n_hidden_layers + 1)])
        self.linear_similarity = nn.ModuleList(
            [nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE) for _ in range(config.n_hidden_layers)] + [
                nn.Linear(BERT_HIDDEN_SIZE, 1)])

    def forward(self, input_ids, attention_mask,task_id):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        bert_output = self.bert(input_ids, attention_mask)
        cls_embeddings = bert_output['pooler_output']
        return cls_embeddings

    def get_sim_embeddings(self, input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2, task_id):
        '''Given a batch of pairs of sentences, get the BERT embeddings.'''
        # Step 0: Get [SEP] token ids
        sep_token_id = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.long, device=input_ids_1.device)
        batch_sep_token_id = sep_token_id.repeat(input_ids_1.shape[0], 1)
        # Step 1: Concatenate the two sentences in: sent1 [SEP] sent2 [SEP]
        input_id = torch.cat((input_ids_1, batch_sep_token_id, input_ids_2, batch_sep_token_id), dim=1)
        attention_mask = torch.cat((attention_mask_1, torch.ones_like(batch_sep_token_id), attention_mask_2, torch.ones_like(batch_sep_token_id)), dim=1)
        # Step 2: Get the BERT embeddings
        x = self.forward(input_id, attention_mask, task_id=task_id)
        return x

    def tasklayer_sentiment(self,x):
        for i in range(len(self.linear_sentiment) - 1):
            x = self.linear_sentiment[i](x)
            x = self.linear_sentiment[i](x)
            x = F.relu(x)
        x = self.dropout_sentiment[-1](x)
        logits = self.linear_sentiment[-1](x)
        return logits

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        embeds = self.forward(input_ids, attention_mask, task_id=1)
        y = self.tasklayer_sentiment(embeds)
        return y

    def tasklayer_paraphrase(self,x):
        for i in range(len(self.linear_paraphrase) - 1):
            x = self.dropout_paraphrase[i](x)
            x = self.linear_paraphrase[i](x)
            x = F.relu(x)
        x = self.dropout_paraphrase[-1](x)
        logits = self.linear_paraphrase[-1](x)
        return logits

    def predict_paraphrase(self, input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        crossembs = self.get_sim_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,task_id=1)
        y = self.tasklayer_paraphrase(crossembs)
        return y

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
        embds = self.get_sim_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,task_id=2)
        y = self.tasklayer_similarity(embds)
        return y

def batchloss_sentiment(batch, sharedspace, args,gradient_accs=1):
    b_ids, b_mask, b_labels = (batch['token_ids'],
                               batch['attention_mask'], batch['labels'])
    model, scaler,device = sharedspace.model, sharedspace.scaler, sharedspace.device
    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)
    b_labels = b_labels.to(device)
    embeddings = model.forward(b_ids, b_mask,1)
    logits = model.tasklayer_sentiment(embeddings)
    if args.mul_bentropy:
        multi_labels = torch.zeros(len(logits), args.num_labels).to(device)
        # Scatter 1s in the corresponding positions of single_labels
        multi_labels[range(len(logits)), b_labels.view(-1)] = 1
        loss = F.binary_cross_entropy_with_logits(logits, multi_labels)
    else:
        loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')
    if args.mark_regularizer:
        loss += smart_regularizer(args.smart_weight,embeddings, logits,model.tasklayer_sentiment)
    if not  args.gradient_surgery:
        loss = loss / gradient_accs
        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
    torch.cuda.empty_cache()
    return loss

def batchloss_paraphrase(batch, sharedspace, args,gradient_accs=1):
    input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, b_labels = (
        batch['token_ids_1'], batch['attention_mask_1'],
        batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
    model, scaler, device = sharedspace.model, sharedspace.scaler, sharedspace.device
    input_ids_1 = input_ids_1.to(device)
    attention_mask_1 = attention_mask_1.to(device)
    input_ids_2 = input_ids_2.to(device)
    attention_mask_2 = attention_mask_2.to(device)
    b_labels = b_labels.to(device)
    embeddings = model.get_sim_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,2)
    logits = model.tasklayer_paraphrase(embeddings)
    loss = F.binary_cross_entropy_with_logits(logits.view(-1), b_labels.float(),
                                              reduction='sum') / len(input_ids_2)
    if args.mark_regularizer:
        loss += smart_regularizer(args.smart_weight,embeddings, logits,model.tasklayer_paraphrase)
    if not args.gradient_surgery:
        loss = loss / gradient_accs
        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
    torch.cuda.empty_cache()
    return loss

def batchloss_similarity(batch, sharedspace, args,gradient_accs=1):
    input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, b_labels = (
        batch['token_ids_1'], batch['attention_mask_1'],
        batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
    model, scaler, device = sharedspace.model, sharedspace.scaler, sharedspace.device
    input_ids_1 = input_ids_1.to(device)
    attention_mask_1 = attention_mask_1.to(device)
    input_ids_2 = input_ids_2.to(device)
    attention_mask_2 = attention_mask_2.to(device)
    b_labels = b_labels.to(device)
    embeddings = model.get_sim_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,3)
    preds = model.tasklayer_similarity(embeddings)
    loss = F.mse_loss(preds.view(-1), b_labels.view(-1), reduction='sum') / len(input_ids_2)
    if args.mark_regularizer:
        loss += smart_regularizer(args.smart_weight,embeddings, preds,model.tasklayer_similarity,type='reg')
    if not args.gradient_surgery :
        loss = loss/gradient_accs
        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
    torch.cuda.empty_cache() 
    return loss

def step_optimizer(sharedspace, args: dict, step: int, total_nb_batches = None):
    """Step the optimizer and update the scaler. Returns the loss"""
    optimizer, scaler = sharedspace.optimizer, sharedspace.scaler
    if args.use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad()
    loss_value = sharedspace.loss_sum
    sharedspace.loss_sum = 0
    torch.cuda.empty_cache()
    return loss_value

class BatchSch:
    '''A class for sampling scheduler.'''

    def __init__(self, dataloaders, reset=True):
        self.dataloaders = dataloaders
        self.names = list(dataloaders.keys())
        if reset: self.reset()

    def reset(self):
        self.sst_iter = iter(self.dataloaders['sst'])
        self.para_iter = iter(self.dataloaders['para'])
        self.sts_iter = iter(self.dataloaders['sts'])
        self.steps = {'sst': 0, 'para': 0, 'sts': 0}
        self.index = 0

    def nextSST_batch(self):
        try:
            return next(self.sst_iter)
        except StopIteration:
            self.sst_iter = cycle(self.dataloaders['sst'])
            return next(self.sst_iter)

    def nextParaphrase_batch(self):
        try:
            return next(self.para_iter)
        except StopIteration:
            self.para_iter = cycle(self.dataloaders['para'])
            return next(self.para_iter)

    def nextSTS_batch(self):
        try:
            return next(self.sts_iter)
        except StopIteration:
            self.sts_iter = cycle(self.dataloaders['sts'])
            return next(self.sts_iter)

    def get_batch(self, name: str):
        if name == "sst":
            return self.nextSST_batch()
        elif name == "para":
            return self.nextParaphrase_batch()
        elif name == "sts":
            return self.nextSTS_batch()
        raise ValueError(f"Unknown batch name: {name}")

    def process_named_batch(self, sharedspace, args, name, apply_optimization = True):
        '''Processes one batch of data from the given dataset, and updates the model accordingly.'''
        batch_fn = None
        if name == "sst":
            batch_fn = batchloss_sentiment
            gradient_accs = args.gradient_acc_sst
        elif name == "para":
            batch_fn = batchloss_paraphrase
            gradient_accs = args.gradient_acc_para
        elif name == "sts":
            batch_fn = batchloss_similarity
            gradient_accs = args.gradient_acc_sts
        else:
            raise ValueError(f"Unknown batch name: {name}")
        loss_of_batch = 0
        for _ in range(gradient_accs):
            batch = self.get_batch(name)
            loss_of_batch += batch_fn(batch, sharedspace, args,gradient_accs)
            torch.cuda.empty_cache()
            gc.collect() 
        if args.gradient_surgery:
            loss_of_batch /= gradient_accs
        # Update the model
        self.steps[name] += 1
        if apply_optimization:
            step_optimizer(sharedspace, args, step=self.steps[name])
        return loss_of_batch

    def round_processbatch(self):
        name = self.names[self.index]
        self.index = (self.index + 1) % len(self.names)
        return name

    def random_processbatch(self):
        name = random.choice(self.names)
        self.index = self.names.index(name)
        return name


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(), # 'optim': optimizer.optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    if args.use_gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.use_gpu}")
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train', dataug=args.data_aug,test=args.test_mode)
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='validation',dataug=args.data_aug,test=args.test_mode)

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size_para,
                                       collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size_para,
                                     collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size_sts,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size_sts,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'n_hidden_layers':args.n_hidden_layers,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    if args.gradient_surgery:
        optimizer = PCGrad(AdamW(model.parameters(), lr=lr))
    else:
        optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0
    scaler = None if not args.use_amp else GradScaler()
    sharedspace= SimpleNamespace(model= model, optimizer=optimizer, scaler=scaler,device=device)
    dataloaders = {'sst': sst_train_dataloader, 'para': para_train_dataloader, 'sts': sts_train_dataloader}
    batch_scheduler=BatchSch(dataloaders)

    clip_value=1
    for p in model.parameters():
        if p.requires_grad:
            if p.requires_grad:p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
    # Run for the specified number of epochs.
    n_batches = 0
    total_num_batches = {'sst': 0, 'para': 0, 'sts': 0}
    num_batches_per_epoch = args.num_batches_per_epoch
    if num_batches_per_epoch <= 0:
        num_batches_per_epoch = int(len(sst_train_dataloader) / args.gradient_acc_sst) + \
                                int(len(para_train_dataloader) / args.gradient_acc_para) + \
                                int(len(sts_train_dataloader) / args.gradient_acc_sts)
    train_loss_logs_epochs = {'sst': [], 'para': [], 'sts': []}
    train_acc_logs_epochs = {'total': [], 'sst': [], 'para': [], 'sts': []}
    dev_acc_logs_epochs = {'total':[],'sst': [], 'para': [], 'sts': []}
    best_dev_acc = -999
    best_dev_accuracies = None
    for epoch in range(args.epochs):
        model.train()
        train_loss = {'sst': 0, 'para': 0, 'sts': 0}
        num_batches = {'sst': 0, 'para': 0, 'sts': 0}
        if args.gradient_surgery:
            for i in tqdm(range(num_batches_per_epoch // 3), desc=f'Train {epoch}', disable=TQDM_DISABLE,
                          smoothing=0):
                losses = []
                for j, name in enumerate(['sst', 'sts', 'para']):
                    losses.append(batch_scheduler.process_named_batch(sharedspace, args=args, name=name,
                                                                apply_optimization=False))
                    n_batches += 1
                    train_loss[name] += losses[-1].item()
                    num_batches[name] += 1
                    total_num_batches[name] += 1
                optimizer.pc_backward(losses)
                optimizer.step()
                torch.cuda.empty_cache()
        else:
            for i in tqdm(range(num_batches_per_epoch), desc=f'Train {epoch}', disable=TQDM_DISABLE, smoothing=0):
                task = batch_scheduler.round_processbatch()
                loss = batch_scheduler.process_named_batch(sharedspace, args=args, name=task,
                                                                apply_optimization=False)
                n_batches += 1
                train_loss[task] += loss.item()
                num_batches[task] += 1
                total_num_batches[task] += 1
                torch.cuda.empty_cache()

        for task in train_loss:
            train_loss[task] = 0 if num_batches[task] == 0 else train_loss[task] / num_batches[task]
            train_loss_logs_epochs[task].append(train_loss[task])

        # Eval dev data
        (sentiment_accuracy_train, _, _,
         paraphrase_accuracy_train, _, _,
         sts_corr_train, _, _) = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model,
                                                device)

        (sentiment_accuracy_dev, _, _,
         paraphrase_accuracy_dev, _, _,
         sts_corr_dev, _, _) = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model,
                                                device)

        arithmetic_mean_train = (paraphrase_accuracy_train + sentiment_accuracy_train + sts_corr_train) / 3
        arithmetic_mean_dev = (paraphrase_accuracy_dev + sentiment_accuracy_dev + sts_corr_dev) / 3

        print(f"Epoch {epoch}: sst train loss :  {train_loss['sst'] :.3f}, train acc :: {sentiment_accuracy_train :.3f}, dev acc :: {sentiment_accuracy_dev :.3f}")
        print(f"Epoch {epoch}: para train loss :  {train_loss['para'] :.3f}, train acc :: {paraphrase_accuracy_train :.3f}, dev acc :: {paraphrase_accuracy_dev :.3f}")
        print(f"Epoch {epoch}: sts train loss :  {train_loss['sst'] :.3f}, train corr :: {sts_corr_train :.3f}, dev corr :: {sts_corr_dev :.3f}")

        print(f"Epoch {epoch}: total train loss :  {np.mean(list(train_loss.values())):.3f}, train acc :: {arithmetic_mean_train :.3f}, dev acc :: {arithmetic_mean_dev :.3f}")
        if arithmetic_mean_dev > best_dev_acc:
               best_dev_acc = arithmetic_mean_dev
               best_dev_accuracies = {'arithmetic_mean_dev':arithmetic_mean_dev,'sst': sentiment_accuracy_dev,
                                      'para': paraphrase_accuracy_dev, 'sts': sts_corr_dev}
               saved_path = save_model(model, optimizer, args, config, args.filepath)

        train_acc_logs_epochs['total'].append(arithmetic_mean_train)
        train_acc_logs_epochs['sst'].append(sentiment_accuracy_train)
        train_acc_logs_epochs['para'].append(paraphrase_accuracy_train)
        train_acc_logs_epochs['sts'].append(sts_corr_train)

        dev_acc_logs_epochs['total'].append(arithmetic_mean_dev)
        dev_acc_logs_epochs['sst'].append(sentiment_accuracy_dev)
        dev_acc_logs_epochs['para'].append(paraphrase_accuracy_dev)
        dev_acc_logs_epochs['sts'].append(sts_corr_dev)
    outfl = 'trainloss' + ('' if args.dataid is None else '_' + args.dataid) + '.csv'
    pd.DataFrame(train_loss_logs_epochs).to_csv(args.tarpath + '/' + outfl)
    outfl = 'trainacc' + ('' if args.dataid is None else '_' + args.dataid) + '.csv'
    pd.DataFrame(train_acc_logs_epochs).to_csv(args.tarpath + '/' + outfl)
    outfl = 'devacc' + ('' if args.dataid is None else '_' + args.dataid) + '.csv'
    pd.DataFrame(dev_acc_logs_epochs).to_csv(args.tarpath + '/' + outfl)
    outfl = 'bestacc' + ('' if args.dataid is None else '_' + args.dataid) + '.csv'
    pd.DataFrame(best_dev_accuracies,index=['dev']).to_csv(args.tarpath + '/' + outfl)
def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        if args.use_gpu < 0:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{args.use_gpu}")
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", type=int, default=0)
    parser.add_argument("--batch_size", help='This is the simulated batch size using gradient accumulations', type=int,
                        default=64)
    parser.add_argument("--num_batches_per_epoch", type=int, default=-1)
    parser.add_argument("--max_batch_size_sst", type=int, default=16)
    parser.add_argument("--max_batch_size_para", type=int, default=8)
    parser.add_argument("--max_batch_size_sts", type=int, default=16)
    parser.add_argument("--patience", type=int, help="Number maximum of epochs without improvement", default=5)
    parser.add_argument("--smart_weight", type=float, default=2e-2)
    parser.add_argument("--mark_regularizer", action='store_true')
    parser.add_argument("--n_hidden_layers", type=int, default=2, help="Number of hidden layers for the classifier")

    parser.add_argument("--gradient_surgery",  action='store_true')
    parser.add_argument("--use_amp", action='store_true')
    parser.add_argument("--test_mode",  action='store_true')
    
    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")
 
    parser.add_argument("--no_train_cpal", action='store_true')
    parser.add_argument("--data_aug", action='store_true')
    parser.add_argument("--mul_bentropy", action='store_true')
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--root", type=str, help="root path for training",
                        default='/adt_geocomplex/bertresult_multitasks_baseline')
    args = parser.parse_args()

    args.gradient_acc_sst = int(np.ceil(args.batch_size / args.max_batch_size_sst))
    args.batch_size_sst = args.batch_size // args.gradient_acc_sst
    args.gradient_acc_para = int(np.ceil(args.batch_size / args.max_batch_size_para))
    args.batch_size_para = args.batch_size // args.gradient_acc_para
    args.gradient_acc_sts = int(np.ceil(args.batch_size / args.max_batch_size_sts))
    args.batch_size_sts = args.batch_size // args.gradient_acc_sts
    args.num_labels = 5 
    return args


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    args.tarpath = retrievePathold(args)
    args.sst_dev_out=args.tarpath+'/'+args.sst_dev_out
    args.sst_test_out=args.tarpath+'/'+args.sst_test_out
    args.para_dev_out=args.tarpath+'/'+args.para_dev_out
    args.para_test_out=args.tarpath+'/'+args.para_test_out
    args.sts_dev_out=args.tarpath+'/'+args.sts_dev_out
    args.sts_test_out=args.tarpath+'/'+args.sts_test_out
    args.filepath =args.tarpath + '/multitasks_classifier.pt' # Save path.
    args.dataid = 'combined'
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    # Training ... ...
    train_multitask(args)
    # Testing ... ...
    test_multitask(args)
