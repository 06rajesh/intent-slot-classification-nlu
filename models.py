import os

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig

from configs import Config

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class JointBert(BertPreTrainedModel):
    def __init__(self, config:BertConfig, args: Config, bert_model="bert-base-uncased"):
        super(JointBert, self).__init__(config)

        self.args = args
        self.bert = BertModel.from_pretrained(bert_model)

        self.num_intent_class = len(args.intents_list)
        self.num_slot_labels = len(args.tags_list)

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_class, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

    def soft_load_from_pretrained(self, model_path:str, device:torch.device = 'cpu'):
        if not os.path.isfile(model_path):
            raise Warning('File not found. Model could not be loaded from pretrained')

        checkpoint = torch.load(model_path, map_location=device)
        loaded_model = checkpoint['model']
        model_dict = self.state_dict()

        missed_layer = []
        for key in checkpoint['model'].keys():
            if key in model_dict and model_dict[key].shape == loaded_model[key].shape:
                pname = key
                pval = loaded_model[key]
                model_dict[pname] = pval.clone().to(model_dict[pname].device)
            else:
                print(key, model_dict[key].shape, loaded_model[key].shape)
                missed_layer.append(key)

        if len(missed_layer) > 0:
            print('{} layers has mismatched shape. Could not be loaded. Following layers to be fine-tuned'.format(
                len(missed_layer)))
            print(missed_layer)
        else:
            print('All layers loaded successfully')

        self.load_state_dict(model_dict)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_class == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_class), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            slot_loss_fct = nn.CrossEntropyLoss()
            slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits