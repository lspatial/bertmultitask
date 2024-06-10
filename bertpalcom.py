import torch
import torch.nn as nn
import torch.nn.functional as F
from bert import BertModel,BertLayer,BertSelfAttention
from utils import *

class TaskSpecificAttention(nn.Module):
  def __init__(self, config, project_up = None, project_down = None, perform_initial_init=False):
    super().__init__()
    #print("Creating project down layer with hidden size", config.hidden_size, "and low rank size", config.low_rank_size)
    self.project_down = nn.Linear(config.hidden_size, config.low_rank_size) if project_down is None else project_down
    self.project_up = nn.Linear(config.low_rank_size, config.hidden_size) if project_up is None else project_up
    config_self_attention = copy.deepcopy(config)
    config_self_attention.hidden_size = config.low_rank_size
    # config_self_attention.num_attention_heads = 6
    self.attention = BertSelfAttention(config_self_attention, init_to_identity=perform_initial_init)

    # Intialize the weight of project_down, project_up such that the self-attention is the zero function
    if perform_initial_init:
      if project_down is None:
        self.project_down.weight.data.zero_()
        self.project_down.bias.data.zero_()
      if project_up is None:
        self.project_up.weight.data.zero_()
        self.project_up.bias.data.zero_()


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # Step 1: project to a lower rank space
    #print("Project down shape", self.project_down
    #print("hidden_states", hidden_states.shape)
    low_rank_hidden_states = self.project_down(hidden_states)
    #print("low_rank_hidden_states", low_rank_hidden_states.shape)
    low_rank_attention_mask = attention_mask

    # Step 2: apply the original BERT self-attention layer
    attn_value = self.attention(low_rank_hidden_states, low_rank_attention_mask)
    #print("attn_value", attn_value.shape)

    # Step 3: project back to the original hidden size
    attn_value = self.project_up(attn_value)

    return attn_value

class BertLayerWithPAL(BertLayer):
    def __init__(self, config, project_ups=None, project_downs=None,train_pal=True):
        super().__init__(config)
        self.pallayersinit(config, project_ups, project_downs, train_pal)
        #Task shared attention

    def pallayersinit(self,config, project_ups=None, project_downs=None,device=None,train_pal=True):
        self.common_up = nn.Linear(config.low_rank_size, config.hidden_size)
        self.down_up = nn.Linear(config.hidden_size, config.low_rank_size)
        self.task_shared_attention = TaskSpecificAttention(config, project_up=self.common_up,
                                                      project_down=self.down_up)
        # Task-specific attention
        self.project_ups = nn.ModuleList([nn.Linear(config.low_rank_size, config.hidden_size) for task in
                                          range(config.num_tasks)]) if project_ups is None else project_ups
        self.project_downs = nn.ModuleList([nn.Linear(config.hidden_size, config.low_rank_size) for task in
                                            range(config.num_tasks)]) if project_downs is None else project_downs
        self.task_attention = nn.ModuleList(
            [TaskSpecificAttention(config, project_up=self.project_ups[task], project_down=self.project_downs[task]) for
             task in range(config.num_tasks)])
        nn.init.zeros_(self.common_up.weight)
        nn.init.zeros_(self.down_up.weight)
        for param in self.task_shared_attention.parameters():
            param.requires_grad = train_pal
        for project_up in self.project_ups:
            nn.init.zeros_(project_up.weight)
        for project_down in self.project_downs:
            nn.init.zeros_(project_down.weight)
        for param in self.task_attention.parameters():
            param.requires_grad = train_pal

    def forward(self, hidden_states, attention_mask, task_id):
        """
        hidden_states: [bs, seq_len, hidden_state]
        attention_mask: [bs, 1, 1, seq_len]
        task_id: int
        output: [bs, seq_len, hidden_state]
        """
        self_attention_output = self.self_attention(hidden_states, attention_mask)
        task_attention_output = self.task_attention[task_id](hidden_states, attention_mask)
        task_shared_output = self.task_shared_attention(hidden_states, attention_mask)

        attention_output = self_attention_output + task_attention_output + task_shared_output
        self_attention_output = self.add_norm(hidden_states, attention_output, self.attention_dense,
                                              self.attention_dropout, self.attention_layer_norm)
        interm_output = self.interm_af(self.interm_dense(self_attention_output))
        output = self.add_norm(self_attention_output, interm_output, self.out_dense, self.out_dropout,
                               self.out_layer_norm)
        # print("output", output.shape)
        return output

    def from_BertLayer(bert_layer, config,device, train_pal=True):
        """
        this function is used to convert a BertLayer to BertLayerWithPAL
        bert_layer: BertLayer
        config: BertConfig
        output: BertLayerWithPAL
        """
        # Hint: you can use the following code to convert a BertLayer to BertLayerWithPAL
        # pal_layer = BertLayerWithPAL.from_BertLayer(bert_layer, config)
        bert_layer.__class__ = BertLayerWithPAL
        # print(config.low_rank_size)
        bert_layer.pallayersinit(config, None, None,device, train_pal)
        return bert_layer


class BertModelWithPAL(BertModel):
  def __init__(self, config):
    super().__init__(config, bert_layer=BertLayerWithPAL)

  def from_BertModel(bert_model, bert_config,device, train_pal=True):
    bert_model.__class__ = BertModelWithPAL
    bert_model.bert_layers = nn.ModuleList([BertLayerWithPAL.from_BertLayer(bert_layer, bert_config, device, train_pal) for bert_layer in bert_model.bert_layers])

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
      # feed the encoding from the last bert_layer to the next
      #print("Encode layer: ", i, " task_id: ", task_id, " hidden_states: ", hidden_states.shape, " attention_mask: ", extended_attention_mask.shape)
      hidden_states = layer_module(hidden_states, extended_attention_mask, task_id=task_id)
      #print("hidden_states after layer: ", hidden_states.shape)

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
