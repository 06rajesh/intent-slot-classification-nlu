from pathlib import Path
from transformers import BertTokenizer, BertConfig
from typing import List, Optional

import torch

from configs import Config
from models import JointBert


class Predictor(object):
    args: Config
    model: JointBert
    tokenizer: BertTokenizer

    _intent_list:List[str]
    _slot_list:List[str]

    def __init__(self, args:Config):
        configuration = BertConfig()
        self.model = JointBert(config=configuration, args=args)
        self.model.soft_load_from_pretrained(model_path=str(model_path))

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_name, model_max_length=args.max_sentence_length)

        self._intent_list = args.intents_list
        self._slot_list = args.tags_list

    def intent_prediction_to_label(self, predictions: torch.tensor):
        intents = []
        for i, x in enumerate(predictions):
            intents.append(self._intent_list[x.item()])
        return intents

    def slot_prediction_to_label(self, predictions: torch.tensor):
        sentences = []
        for i, s in enumerate(predictions):
            sentence = []
            for j, w in enumerate(s):
                slot_idx = w.item()
                if slot_idx != 0:
                    sentence.append(self._slot_list[slot_idx])
            sentences.append(sentence)
        return sentences

    def clean_output_tokens(self, slot_token: str):
        slot_token = slot_token.replace(" ##", "")

        return slot_token

    def _convert_sentence_to_output_format(self, sentence: str, intent: str, slots: List[str]):
        tokens = self.tokenizer.tokenize(sentence)

        assert len(tokens) == len(slots)

        slot_val = list()
        slot_key = list()

        slot_token = ""
        current_slot = ""
        for i, slot in enumerate(slots):
            if slot != 'O':
                label = slot[2:]
                if label == current_slot:
                    slot_token += " " + tokens[i]
                else:
                    if slot_token != "" and current_slot != "":
                        slot_val.append(slot_token)
                        slot_key.append(current_slot)
                    slot_token = tokens[i]
                    current_slot = label
            elif slot_token != "" and current_slot != "":
                slot_val.append(slot_token)
                slot_key.append(current_slot)
                slot_token = ""
                current_slot = ""

        if slot_token != "" and current_slot != "":
            slot_val.append(slot_token)
            slot_key.append(current_slot)

        slot_dict = dict()
        for s in range(len(slot_val)):
            slot_dict[slot_key[s].lower()] = self.clean_output_tokens(slot_val[s])

        return {
            "intent": intent,
            "text": sentence,
            "slots": slot_dict,
        }

    @torch.no_grad()
    def predict(self, sentences: List[str]):

        self.model.eval()

        inputs = self.tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        inputs.update({
            "intent_label_ids": None,
            "slot_labels_ids": None,
        })

        outputs = self.model(**inputs)
        losses, (intent_logits, slot_logits) = outputs

        intent_preds = torch.argmax(intent_logits.detach(), dim=1)
        intents = self.intent_prediction_to_label(intent_preds)

        slot_preds = torch.argmax(slot_logits.detach(), dim=2)
        slots = self.slot_prediction_to_label(slot_preds)

        for i, s in enumerate(sentences):
            formatted = self._convert_sentence_to_output_format(s, intents[i], slots[i])
            print(formatted)

if __name__ == '__main__':
    output_dir = Path('./pretrained/v.0.2')

    args = Config.from_pretrained(str(output_dir))
    args.device = 'cpu'
    args.batch_size = 4
    args.output_dir = output_dir

    model_path = output_dir / 'checkpoint.pth'

    # print(args.__dict__)

    sentences = [
        "I'm looking for a local cafeteria that has wifi accesss for a party of 4",
        "book for one in Indiana at a restaurant",
        "Add As I Was Going to St Ives to the fantasía playlist."
    ]

    predictor = Predictor(args)
    predictor.predict(sentences)