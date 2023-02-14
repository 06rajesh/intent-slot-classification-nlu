import json
import pickle

from typing import Optional
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

import torch
import torch.distributed as dist

from transformers import BertTokenizer, PreTrainedTokenizer
from unidecode import unidecode

class IntentClassifier:
    def __init__(self, intent_labels=[]):
        self.clf = self._create_model()
        self.labels = intent_labels
        self._savepath = "models/intentclf.pickle"

    @staticmethod
    def _create_model():
        clf = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', SGDClassifier())
                        ])
        return clf

    def save(self):
        outfile = open(self._savepath, 'wb')
        pickle.dump([self.clf, self.labels], outfile)
        outfile.close()

    def load(self):
        infile = open(self._savepath, 'rb')
        clf, labels = pickle.load(infile)
        self.clf = clf
        self.labels = labels
        infile.close()

    def get_intent_labels(self):
        return self.labels

    def train(self, train_x, train_y):
        self.clf = self.clf.fit(train_x, train_y)

    def predict(self, test_x):
        return self.clf.predict(test_x)


class OneHotEncoder:
    def __init__(self, classes: list):
        self.classes = classes

    def encode(self, outputs):
        encoded_outputs = []
        for row in outputs:
            encoded_row = [0] * len(self.classes)
            for idx, label in enumerate(self.classes):
                if label in row:
                    encoded_row[idx] = 1
            encoded_outputs.append(encoded_row)
        return encoded_outputs


class WordLabelMapper:
    tokenizer: Optional[BertTokenizer]
    spcl_char: str
    unk_token: str

    def __init__(self, tokenizer:PreTrainedTokenizer=None):
        self.tokenizer = tokenizer
        self.spcl_char = "#"
        self.unk_token = "[UNK]"

    @staticmethod
    def verify_strings(str1, str2) -> bool:
        if str1 == str2:
            return True
        else:
            str2 = unidecode(str2)
            if str1 == str2:
                return True
        return False

    def tokenize_word_with_limits(self, sentence:str)-> (list, list):
        tokenized = self.tokenizer.tokenize(sentence)

        copied = sentence.lower()
        words = list()
        limits = list()
        start: int = 0
        for token in tokenized:
            clean_token = token.replace(self.spcl_char, "")
            c = len(clean_token)

            # unknown token should be of 1 length for all cases
            if clean_token == self.unk_token:
                c = 1

            idx = 0
            for i in range(len(copied)):
                if copied[i] == " ":
                    continue
                else:
                    idx = i
                    break

            token_start = idx
            token_end = idx + c

            # no need to verify in case of unknown token
            if clean_token != self.unk_token:
                verified = self.verify_strings(clean_token, copied[token_start:token_end])
            else:
                verified = True

            if verified:
                limits.append([start+token_start, start+token_end - 1])
                words.append(token)
                copied = copied[token_end:]
                start += token_end
            else:
                print(sentence)
                print(tokenized)
                print(token)
                print(copied)
                raise ValueError('tokenization error.')

        assert len(words) == len(limits)

        return words, limits

    @staticmethod
    def get_word_with_limits(sentence: str) -> (list, list):
        words = list()
        limits = list()
        start: int = 0

        for c in range(0, len(sentence)):
            if sentence[c] == " ":
                word = sentence[start:c]
                words.append(word)
                limits.append([start, c - 1])
                start = c + 1
            elif c == len(sentence) - 1:
                word = sentence[start:c+1]
                words.append(word)
                limits.append([start, c])
        return words, limits

    @staticmethod
    def get_word_limits(sentence: str) -> list:
        limits = list()
        start: int = 0
        for c in range(0, len(sentence)):
            if sentence[c] == " ":
                limits.append([start, c-1])
                start = c + 1
            elif c == len(sentence)-1:
                limits.append([start, c])
        return limits

    @staticmethod
    def check_word_limit_with_label(limit: list, labels: dict) -> str:
        for key in labels.keys():
            position = labels[key]
            if len(position) > 1 and len(limit) > 1:
                if limit[0] == position[0] and limit[1] == position[1]:
                    return 'B-' + key.upper()
                elif limit[0] == position[0] and limit[1] < position[1]:
                    return 'B-' + key.upper()
                elif limit[0] > position[0] and limit[1] <= position[1]:
                    return 'I-' + key.upper()
        return 'O'

    def compute_sentence_labels(self, sentence: str, labels: dict) -> (list, list):
        """
        separate words using space and label each word from labels dictionary
        where all slots with positions are in the dictionary
        :param sentence: string with many words
        :param labels: dictionary of labels with their respective positions
        :return: list of words and respective tags on a different list
        """
        if self.tokenizer:
            words, limits = self.tokenize_word_with_limits(sentence)
        else:
            words, limits = self.get_word_with_limits(sentence)

        new_words = list()
        tags = list()
        for i in range(len(limits)):
            if words[i] != '':
                label = self.check_word_limit_with_label(limits[i], labels)
                new_words.append(words[i])
                tags.append(label)
        return new_words, tags


def import_data(file_path:str, limit=-1, tokenizer:PreTrainedTokenizer=None):
    with open(file_path, "r") as read_file:
        json_data = json.load(read_file)

    mapper = WordLabelMapper(tokenizer=tokenizer)
    sentences, labels = list(), list()
    idx = 0

    max_sentence_len = 0
    for key in json_data.keys():
        data = json_data[key]
        words, tagged = mapper.compute_sentence_labels(data['text'], data['positions'])

        intent = data['intent']

        if len(words) > max_sentence_len:
            max_sentence_len = len(words)

        sentences.append(words)
        labels.append((intent, tagged))
        if limit != -1 and idx >= limit:
            break
        idx += 1

    print('Imported data with a maximum sentence length of: {}'.format(max_sentence_len))
    return sentences, labels


def import_intent_data(file_path: str, pre_intent_labels: list = None):
    with open(file_path, "r") as read_file:
        json_data = json.load(read_file)

    sentences = list()
    intent = list()
    intent_labels = set()
    for key in json_data.keys():
        data = json_data[key]
        sentences.append(data["text"])
        label = data['intent']
        intent.append(label)
        intent_labels.add(label)

    if pre_intent_labels is not None:
        intent_labels = pre_intent_labels
    else:
        intent_labels = list(intent_labels)
    intent_trans = [intent_labels.index(l) for l in intent]

    return sentences, intent_trans, intent_labels


def import_dev_data(file_path: str):
    with open(file_path, "r") as read_file:
        json_data = json.load(read_file)

    sentences = list()
    for key in json_data.keys():
        data = json_data[key]
        sentence = data["text"]
        # tokens = sentence.split()
        sentences.append(sentence)

    return sentences

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)