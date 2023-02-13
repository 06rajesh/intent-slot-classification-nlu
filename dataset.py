from typing import List

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, PreTrainedTokenizer

class NLUDataset(Dataset):
    def __init__(self, sentences:List, labels: List[tuple], transform=None):

        self.transform = transform

        self.idx_to_intents = []
        self.intent_to_idx = {}
        self.idx_to_tags = []
        self.tags_to_idx = {}
        self.max_length = 0

        self._inputs = sentences
        self._target_intents = []
        self._target_tags = []
        self._total = len(sentences)

        self._process_data(sentences, labels)

    def _process_data(self, sentences:List, labels: List[tuple]):
        target_intents = []
        target_tags = []
        intents = list()
        # initialize the tag list with PADDING (PAD) class
        tags = ['-PAD-']
        max_len = 0
        for i in range(len(labels)):
            intent = labels[i][0]
            tag_labels = labels[i][1]

            target_intents.append(intent)
            target_tags.append(tag_labels)

            for t in tag_labels:
                if t not in tags:
                    tags.append(t)

            if intent not in intents:
                intents.append(intent)

            if len(sentences[i]) > max_len:
                max_len = len(sentences[i])

        self._target_intents = target_intents
        self._target_tags = target_tags

        intents_to_idx = {c: i for i, c in enumerate(intents)}
        tags_to_idx = {c: i for i, c in enumerate(tags)}

        self.idx_to_intents = intents
        self.intents_to_idx = intents_to_idx

        self.idx_to_tags = tags
        self.tags_to_idx = tags_to_idx

        self.max_length = max_len

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        intent = self.intents_to_idx[self._target_intents[idx]]
        tags = [self.tags_to_idx[tag] for tag in self._target_tags[idx]]

        inputs = self._inputs[idx]
        targets = {
            'intent': intent,
            'tags' : tags,
        }
        sample = inputs, targets

        if self.transform:
            self.transform(sample)

        return sample

    def _to_tensor(self, target):
        return {
            'intent': torch.tensor(target['intent'], dtype=torch.int8),
            'tags': torch.Tensor(target['tags']),
        }

class Collator(object):
    def __init__(self, tokenizer:PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.max_len = tokenizer.model_max_length

    def pad_output(self, output):
        padded = [0] * self.max_len
        for i in range(len(output)):
            padded[i+1] = output[i]

        return padded

    def __call__(self, data):
        sentences = [self.tokenizer.convert_tokens_to_string(input) for (input, target) in data]
        intents = [target['intent'] for (input, target) in data]
        tags = [self.pad_output(target['tags']) for (input, target) in data]

        inputs = self.tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        targets = {
            'intents': torch.tensor(intents),
            'tags': torch.tensor(tags)
        }

        return inputs, targets
