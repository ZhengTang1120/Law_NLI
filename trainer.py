from transformers import AutoConfig, AutoModel, Trainer, PreTrainedModel
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
import torch

class CustomModel(PreTrainedModel):
     def __init__(self, config, num_labels):
        super().__init__(config)
        self.name = config.name_or_path
        self.encoder = AutoModel.from_config(config)
        self.attn = nn.MultiheadAttention(config.hidden_size, 1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        if 'roberta' in self.name:
          self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.loss_fn = nn.CrossEntropyLoss()

     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, task_ids=None, return_dict=False, **kwargs):

          outputs = self.encoder(
               input_ids,
               attention_mask=attention_mask,
               token_type_ids=token_type_ids,
               **kwargs,
          )

          seq_out = outputs[0].transpose(0, 1)
          outs, attns = self.attn(seq_out, seq_out, seq_out)
          output = outs[0]
          # if 'bert' in self.name:
          #      output = outputs[1]
          # elif 'roberta' in self.name:
          #      output = outputs[0][:, 0, :]
          #      output = self.dropout(output)
          #      output = self.dense(output)
          #      output = torch.tanh(output)
          # else:
          #      print ("unsupported model %s"%self.name)
          #      exit()
          output_dropout = self.dropout(output)
          logits = self.classifier(output_dropout)
          
          loss = None
          if labels is not None:
               loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

          if not return_dict:
               output = (logits.view(-1, self.num_labels),)
               return ((loss,) + output) if loss is not None else output

          return SequenceClassifierOutput(
               loss=loss,
               logits=logits,
               attentions=attns,
          )

class CustomTrainer(Trainer):
     def compute_loss(self, model, inputs, return_outputs=False):
          labels = inputs.get("labels")
          # forward pass
          
          if self.is_in_train:
               outputs = model(**inputs, return_dict=True)
               # Attention regularization
               inputs = inputs.input_ids
               attentions = outputs.attentions # use the last layer

               c = (inputs == 30524).nonzero(as_tuple=True)[1]
               l = (inputs == 30523).nonzero(as_tuple=True)[1]
               p = (inputs == 102).nonzero(as_tuple=True)[1]
               mask  = torch.zeros(attentions.size()).cuda()
               mask1 = torch.zeros(inputs.size()).cuda()
               for i in range(inputs.size(0)):
                    mask1[i].index_fill_(0, torch.arange(0, 1).cuda(), 1) # exclude [CLS]
                    mask1[i].index_fill_(0, torch.arange(1, c[i]).cuda(), 1)
                    mask1[i].index_fill_(0, torch.arange(p[i], mask1[i].size(0)).cuda(), 1) # exclude [SEP] and [PAD]
                    mask[i][:c[i]] = mask1[i]
               mask1 = torch.zeros(inputs.size()).cuda()
               for i in range(inputs.size(0)):
                    mask1[i].index_fill_(0, torch.arange(0, 1).cuda(), 1)
                    mask1[i].index_fill_(0, torch.arange(c[i], l[i]).cuda(), 1)
                    mask1[i].index_fill_(0, torch.arange(p[i], mask1[i].size(0)).cuda(), 1)
                    mask[i][c[i]:l[i]] = mask1[i]
               mask1 = torch.zeros(inputs.size()).cuda()
               for i in range(inputs.size(0)):
                    mask1[i].index_fill_(0, torch.arange(0, 1).cuda(), 1) 
                    mask1[i].index_fill_(0, torch.arange(l[i], p[i]).cuda(), 1)
                    mask1[i].index_fill_(0, torch.arange(p[i], mask1[i].size(0)).cuda(), 1)
                    mask[i][l[i]:p[i]] = mask1[i]
               attn_reg = torch.sum(attentions * mask)
               mask1 = mask = None
          else:
               attn_reg = 0
               outputs = model(**inputs)
          if self.is_in_train:
               loss = outputs.get("loss").mean()
          else:
               loss = outputs[0].mean()
          beta = 0.1/(10 ** len(str((attn_reg // loss).item()).split('.')[0]))
          loss += beta * attn_reg
          return (loss, outputs) if return_outputs else loss