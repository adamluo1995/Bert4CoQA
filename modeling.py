from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class BertForCoQA(BertPreTrainedModel):
    def __init__(self,
                 config,
                 output_attentions=False,
                 keep_multihead_output=False):
        super(BertForCoQA, self).__init__(config)
        self.output_attentions = output_attentions
        self.bert = BertModel(config,
                              output_attentions=output_attentions,
                              keep_multihead_output=keep_multihead_output)
        self.qa_outputs_mid = nn.Linear(config.hidden_size, config.hidden_size)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.cls_outputs_mid = nn.Linear(config.hidden_size,
                                         config.hidden_size)
        self.cls_outputs = nn.Linear(config.hidden_size, 4)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                start_positions=None,
                end_positions=None,
                cls_idx=None,
                head_mask=None,
                cls_alpha=1.0):
        outputs = self.bert(input_ids,
                            token_type_ids,
                            attention_mask,
                            output_all_encoded_layers=False,
                            head_mask=head_mask)
        if self.output_attentions:
            all_attentions, sequence_output, cls_outputs = outputs
        else:
            sequence_output, cls_outputs = outputs
        span_logits = self.qa_outputs(
            F.relu(self.qa_outputs_mid(sequence_output)))
        cls_logits = self.cls_outputs(F.relu(
            self.cls_outputs_mid(cls_outputs)))

        start_logits, end_logits = span_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            span_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            cls_loss_fct = CrossEntropyLoss()
            start_loss = span_loss_fct(start_logits, start_positions)
            end_loss = span_loss_fct(end_logits, end_positions)
            cls_loss = cls_loss_fct(cls_logits, cls_idx)
            total_loss = (start_loss + end_loss) / 2 + cls_alpha * cls_loss
            return total_loss
        elif self.output_attentions:
            return all_attentions, start_logits, end_logits, cls_logits
        return start_logits, end_logits, cls_logits