import math
from pathlib import Path

from tqdm import tqdm
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, random_split, RandomSampler, SequentialSampler

from utils import import_data, save_on_master
from configs import Config
from dataset import NLUDataset, Collator


class Trainer(object):
    args:Config
    device:torch.device
    output_dir: Path
    testing:bool

    model:Module
    optimizer: torch.optim.Optimizer

    data_loader_train: DataLoader
    data_loader_val: DataLoader

    _max_validation_loss:float

    def __init__(self, args:Config, model:Module, optimizer: torch.optim.Optimizer, dataset:NLUDataset, collator:Collator, testing=False):
        self.args = args
        self.device = torch.device(args.device)
        self.output_dir = Path(args.output_dir)
        self.testing = testing

        self.model = model
        self.optimizer = optimizer

        self._max_validation_loss = 6.0

        self._init_data_loaders(args, dataset, collator)


    def _init_data_loaders(self, args:Config, dataset: NLUDataset, collator:Collator):
        total = len(dataset)

        train_size = math.ceil(total * 0.8)
        test_size = total - train_size

        dataset_train, dataset_val = random_split(dataset, [train_size, test_size])

        sampler_train = RandomSampler(dataset_train)
        sampler_val = SequentialSampler(dataset_val)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
        batch_sampler_val = torch.utils.data.BatchSampler(sampler_val, args.batch_size, drop_last=False)

        self.data_loader_train = DataLoader(dataset_train, num_workers=args.num_workers,
                                       batch_sampler=batch_sampler_train, collate_fn=collator)
        self.data_loader_val = DataLoader(dataset_val, num_workers=args.num_workers,
                                     batch_sampler=batch_sampler_val, collate_fn=collator)

    def _before_train(self):
        self.model.to(self.device)

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        # train model
        print("Start training")

        # save config before start training
        self.args.save_config()

    def train(self):
        self._before_train()

        for epoch in range(self.args.start_epoch, self.args.epochs):
            stats = self.train_one_epoch(epoch)
            validation_stats = self.evaluate()

            self._after_evaluation(validation_stats, epoch)

            if self.testing:
                break

    def _after_train(self):
        return

    def _after_evaluation(self, stats:dict, epoch:int):
        if self.args.output_dir:
            checkpoint_path = self.output_dir / 'checkpoint.pth'
            if stats['avg_loss'] < self._max_validation_loss:
                self._max_validation_loss = stats['avg_loss']
                save_on_master({'model': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'epoch': epoch,
                                'args': self.args}, checkpoint_path)
                print('model saved on {} after {} epoch with validation loss {:.4f}'.format(checkpoint_path, epoch, self._max_validation_loss))


    def train_one_epoch(self, epoch:int):
        self.model.train()
        print_freq = 2

        loader_desc = 'Epoch [{:d}]: loss = {:.4f}, accuracy (intent = {:.4f}, slots =  {:.4f})'
        train_iterator = tqdm(self.data_loader_train, desc=loader_desc.format(epoch, 0.0, 0.0, 0.0))

        intent_total = 0
        intent_correct = 0

        slot_total = 0
        slot_correct = 0

        total_loss = 0
        completed_batch = 0
        for idx, samples in enumerate(train_iterator, 1):
            inputs, targets = samples

            # data & target
            inputs = inputs.to(self.device)
            targets = {k: v.to(self.device) if type(v) is not str else v for k, v in targets.items()}

            inputs.update({
                "intent_label_ids": targets['intents'],
                "slot_labels_ids": targets['tags'],
            })

            outputs = self.model(**inputs)
            losses, (intent_logits, slot_logits) = outputs
            loss_val = losses.item()
            total_loss += loss_val

            intent_preds = torch.argmax(intent_logits.detach(), dim=1)
            intent_acc = torch.sum(intent_preds == targets['intents'])
            intent_total += targets['intents'].shape[0]
            intent_correct += intent_acc.item()

            slot_preds = torch.argmax(slot_logits.detach(), dim=2)
            slot_acc = torch.sum(slot_preds == targets['tags'])
            slot_total += targets['tags'].shape[0] * targets['tags'].shape[1]
            slot_correct += slot_acc.item()

            # loss backward & optimzer step
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            if idx % print_freq == 0:
                train_iterator.set_description(
                    loader_desc.format(epoch, loss_val, intent_correct / intent_total, intent_correct / slot_total))

            completed_batch += 1

            if self.testing and idx == 10:
                break

        print('Total {}/{} correct intents, {}/{} correct slots trained with a total loss of {:.4f} from {} items'
              .format(intent_correct, intent_total, slot_correct, slot_total, total_loss, completed_batch))

        return {
            'avg_loss': total_loss / completed_batch,
            'intent_acc': intent_correct / intent_total,
            'slot_acc': slot_correct / slot_total,
        }

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        print_freq = 2

        loader_desc = 'Validation: loss = {:.4f}, accuracy (intent = {:.4f}, slots =  {:.4f})'
        evaluation_iterator = tqdm(self.data_loader_val, desc=loader_desc.format(0.0, 0.0, 0.0))

        intent_total = 0
        intent_correct = 0

        slot_total = 0
        slot_correct = 0

        total_loss = 0
        completed_batch = 0
        for idx, samples in enumerate(evaluation_iterator, 1):
            inputs, targets = samples

            # data & target
            inputs = inputs.to(self.device)
            targets = {k: v.to(self.device) if type(v) is not str else v for k, v in targets.items()}

            inputs.update({
                "intent_label_ids": targets['intents'],
                "slot_labels_ids": targets['tags'],
            })

            outputs = self.model(**inputs)
            losses, (intent_logits, slot_logits) = outputs
            loss_val = losses.item()
            total_loss += loss_val

            intent_preds = torch.argmax(intent_logits.detach(), dim=1)
            intent_acc = torch.sum(intent_preds == targets['intents'])
            intent_total += targets['intents'].shape[0]
            intent_correct += intent_acc.item()

            slot_preds = torch.argmax(slot_logits.detach(), dim=2)
            slot_acc = torch.sum(slot_preds == targets['tags'])
            slot_total += targets['tags'].shape[0] * targets['tags'].shape[1]
            slot_correct += slot_acc.item()

            if idx % print_freq == 0:
                evaluation_iterator.set_description(
                    loader_desc.format(loss_val, intent_correct / intent_total, intent_correct / slot_total))

            completed_batch += 1

            if self.testing and idx == 10:
                break

        return {
            'avg_loss': total_loss / completed_batch,
            'intent_acc': intent_correct / intent_total,
            'slot_acc': slot_correct / slot_total,
        }


