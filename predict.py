import os
import argparse
import json
from pathlib import Path
from typing import List

import torch
from transformers import BertTokenizer, BertConfig

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

        model_path = output_dir / 'checkpoint.pth'
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

        output = []
        for i, s in enumerate(sentences):
            formatted = self._convert_sentence_to_output_format(s, intents[i], slots[i])
            output.append(formatted)

        return output

    def predict_from_file(self, input_file:str, output_file:str):
        with open(input_file, "r") as read_file:
            input_data = json.load(read_file)

        batch_size = self.args.batch_size

        keys = list(input_data.keys())
        key_chunks = [keys[x:x+batch_size] for x in range(0, len(keys), batch_size)]

        outputs = dict()
        for chunk in key_chunks:
            sentences = []
            for key in chunk:
                sentences.append(input_data[key]['text'])

            chunk_out = self.predict(sentences)
            for i in range(len(chunk_out)):
                outputs[chunk[i]] = chunk_out[i]

        json_out = json.dumps(outputs, indent=4)
        with open(output_file, 'w') as f:
            f.write(json_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='JointBERT predictor',
        description='Predict Intent and slots from input file sentences using Pretrained JointBERT model',
        epilog='developed by Rajesh Baidya')

    parser.add_argument('-m', '--model', help="Pretrained Model Directory", required=True)
    parser.add_argument('-i', '--input', help="Input file for prediction. (See code for input file format)")
    parser.add_argument('-o', '--output', help="Output file for the prediction.")

    args = parser.parse_args()

    samples = [
        "I'm looking for a local cafeteria that has wifi accesss for a party of 4",
        "book for one in Indiana at a restaurant",
        "Add As I Was Going to St Ives to the fantasía playlist."
    ]

    output_dir = Path(args.model)

    config = Config.from_pretrained(str(output_dir))
    config.device = 'cpu'
    config.output_dir = output_dir

    # print(args.__dict__)
    predictor = Predictor(config)

    if args.input == None:
        output = predictor.predict(samples)
        print(output)
    else:
        assert os.path.isfile(args.input)
        input_file = Path(args.input)

        if args.output == None:
            output_file = input_file.parent / 'output.json'
        else:
            output_file = Path(args.output)

        predictor.predict_from_file(str(input_file), str(output_file))
