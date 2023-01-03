from transformers import BertConfig, AutoModel, Trainer, PreTrainedModel,BertForSequenceClassification
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
import torch

class CustomModel(PreTrainedModel):
     def __init__(self, config, name, num_labels):
        super().__init__(config)
        self.name = name
        self.num_labels = num_labels
        self.config = config
        self.bert = AutoModel.from_pretrained(self.name)
        # self.attn = nn.MultiheadAttention(config.hidden_size, 1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        # if 'roberta' in self.name:
        #   self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.loss_fn = nn.CrossEntropyLoss()

        # self.post_init()


     def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

     def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value

     def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        labels = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):

          outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
          output = outputs[1]
          # seq_out = outputs[0].transpose(0, 1)
          # outs, attns = self.attn(seq_out, seq_out, seq_out)
          # output = outs[0]
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
               loss_fn = nn.CrossEntropyLoss()
               loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

          if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
          # return SequenceClassifierOutput(
          #      loss=loss,
          #      logits=logits,
          #      attentions=attns,
          # )
          return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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
          beta = 1#1/(10 ** len(str((attn_reg // loss).item()).split('.')[0]))
          loss += beta * attn_reg
          return (loss, outputs) if return_outputs else loss
