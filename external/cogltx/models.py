# Original code from:
# Ding et al. (2020), "CogLTX: Applying BERT to Long Texts"
# https://github.com/Sleepychord/CogLTX
# License: MIT
#
# Change in this file:
# - Added comments for clarity
# - Changed the word choice of introspector to judge (according to the paper)
# - Removed unused task-specific class (QAReasoner)
# - Use BERT instead of RoBERTa
# - Judge forward function fixed to solve indexing with mask
# - Reasoner forward function fixed to solve error when feeding inputs to CrossEntropyLoss

import torch
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertConfig, BertModel, BERT_PRETRAINED_MODEL_ARCHIVE_LIST, BertForSequenceClassification

# Judge is designed to choose relevant text blocks from long texts.
class Judge(BertPreTrainedModel):

    config_class = BertConfig
    pretrained_model_archive_list = BERT_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "bert"

    def __init__(self, config):
        super(Judge, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(config.hidden_size, 3)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        outputs = logits
        if labels is not None:
            labels = labels.type_as(logits)
            loss_fct = CrossEntropyLoss()

            _, _, num_labels = logits.shape
            logits_flat = logits.view(-1, num_labels)  
            labels_flat = labels.view(-1).long()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_mask = attention_mask.view(-1) == 1  # 
                active_logits = logits_flat[active_mask]    
                active_labels = labels_flat[active_mask]    
            else:
                active_logits = logits_flat
                active_labels = labels_flat

            loss = loss_fct(active_logits, active_labels)

            outputs = (loss, logits)

        return outputs  # (loss), scores, (hidden_states), (attentions)

# Interface for task-specific reasoner classes.
class Reasoner(object): # Interface

    def export_labels(self, bufs, device):
        raise NotImplementedError
        # return (labels: consistent with forward, crucials: list of list of blks)

    def forward(self, ids, attn_masks=None, type_ids=None, labels=None, **kwargs):
        raise NotImplementedError
        # return (loss, ) if labels is not None else ...

# Reasoners is designed to do the main task (classification).
class ClassificationReasoner(BertForSequenceClassification, Reasoner):
    
    def __init__(self, config):
        super(ClassificationReasoner, self).__init__(config)

    @classmethod
    def export_labels(cls, bufs, device):
        labels = torch.zeros(len(bufs), dtype=torch.long, device=device)
        for i, buf in enumerate(bufs):
            labels[i] = int(buf[0].label)
        return labels, [[b for b in buf if b.blk_type == 0] for buf in bufs]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]
        logits = self.classifier(cls_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(reduction='none')
                labels = labels.squeeze().view(-1).long()
                loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)