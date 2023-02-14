from pathlib import Path
from transformers import BertTokenizer, BertConfig

from utils import import_data
from configs import Config
from dataset import NLUDataset, Collator
from models import JointBert
from trainer import Trainer

def evaluate_on_val(args:Config, model:JointBert, dataset:NLUDataset, collator:Collator):
    trainer = Trainer(
        args=args,
        model=model,
        optimizer=None,
        dataset=dataset,
        collator=collator,
        testing=False
    )

    trainer.evaluate()

if __name__ == '__main__':
    output_dir = Path('./pretrained/v.0.2')

    args = Config.from_pretrained(str(output_dir))
    args.device = 'cpu'
    args.batch_size = 2
    args.output_dir = output_dir

    model_path = output_dir / 'checkpoint.pth'

    print(args.__dict__)

    configuration = BertConfig()
    model = JointBert(config=configuration, args=args)
    model.soft_load_from_pretrained(model_path=str(model_path))

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=args.max_sentence_length)
    sentences, labels = import_data("nlu_traindev/train.json", limit=-1, tokenizer=bert_tokenizer)

    collator = Collator(bert_tokenizer)

    dataset = NLUDataset(sentences, labels)
    evaluate_on_val(args, model, dataset, collator)