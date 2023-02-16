import torch
from transformers import BertTokenizer, BertConfig

from utils import import_data
from configs import Config
from dataset import NLUDataset, Collator
from models import JointBert
from trainer import Trainer


def train(args:Config, dataset:NLUDataset, collator:Collator):
    configuration = BertConfig()
    model = JointBert(config=configuration, args=args)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.learning_rate, weight_decay=args.weight_decay)

    trainer = Trainer(
        args=args,
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        collator=collator,
        testing=True
    )

    trainer.train()



if __name__ == '__main__':
    args = Config()

    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model_name, model_max_length=args.max_sentence_length)
    sentences, labels = import_data("nlu_traindev/train.json", limit=-1, tokenizer=bert_tokenizer)

    collator = Collator(bert_tokenizer)

    dataset = NLUDataset(sentences, labels)
    args.tags_list = dataset.idx_to_tags
    args.intents_list = dataset.idx_to_intents

    train(args, dataset, collator)
