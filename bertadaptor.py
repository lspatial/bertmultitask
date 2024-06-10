import torch
import torch.nn as nn
import torch.nn.functional as F
from bert import BertModel,BertLayer,BertSelfAttention
from utils import *
from tokenizer import BertTokenizer
from bert import BertModel
from config import BertConfig
class Adapter(nn.Module):
    """
    The adapters first project the original
    d-dimensional features into a smaller dimension, m, apply
    a nonlinearity, then project back to d dimensions.
    """
    def __init__(self, size = 6, model_dim = 768):
        super().__init__()
        self.adapter_block = nn.Sequential(
            nn.Linear(model_dim, size),
            nn.ReLU(),
            nn.Linear(size, model_dim)
        )

    def forward(self, x):
        ff_out = self.adapter_block(x)
        # Skip connection
        adapter_out = ff_out + x
        return adapter_out

class Adaptered(nn.Module):
    def __init__(self, orig_layer, size,model_dim,ntask):
        super().__init__()
        self.orig_layer = orig_layer
        self.sharedadapter = Adapter(size,model_dim)

    def forward(self, itask, *x):
        orig_out = self.orig_layer(*x)
        shared = self.sharedadapter(orig_out)
        output = shared   # (self.adapter.forward(orig_out[0].unsqueeze(0))[0],)
        return output

class BertModelWithAdaptor(BertModel):
    def __init__(self,config):
        super().__init__(config)

    def from_BertModel(bert_model,config):
        bert_model.__class__ = BertModelWithAdaptor
        # Freeze the original model parameters
        for params in bert_model.parameters():
            params.requires_grad = False
        # Embed adapter layers into the transformer blocks
        for i in range(len(bert_model.bert_layers)):
            bert_model.bert_layers[i] = Adaptered(bert_model.bert_layers[i], config.low_rank_size,config.hidden_size,3)

    def encode(self, hidden_states, attention_mask, task_id):
        """
        hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # get the extended attention mask for self attention
        # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
        # non-padding tokens with 0 and padding tokens with a large negative number
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)
       # pass the hidden states through the encoder layers
        for i, layer_module in enumerate(self.bert_layers):
          hidden_states = layer_module(task_id,hidden_states, extended_attention_mask)
        return hidden_states

    def forward(self, input_ids, attention_mask, task_id):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # get the embedding for each input token
        embedding_output = self.embed(input_ids=input_ids)
        # feed to a transformer (a stack of BertLayers)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask, task_id=task_id)
        # get cls token hidden state
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)
        return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}

#if __name__ == '__main__':
##    bert = BertModel.from_pretrained('bert-base-uncased')
 #   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
 #   config = BertConfig()
 #   BertModelWithAdaptor.from_BertModel(bert,config)
 #   print(bert)
