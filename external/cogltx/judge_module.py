# Original code from:
# Ding et al. (2020), "CogLTX: Applying BERT to Long Texts"
# https://github.com/Sleepychord/CogLTX
# License: MIT
#
# Change in this file:
# - Changed file name from introspector_module.py to judge_module.py -
# - Added comments for clarity
# - Changed the word choice of introspector to judge (according to the paper)
# - Does not use distributed system

import os
import json
import logging
import random
from argparse import ArgumentParser
from copy import deepcopy

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from external.cogltx.optimization import WarmupLinearLR
from external.cogltx.models import Judge
from external.cogltx.buffer import buffer_collate
from external.cogltx.memrecall import _score_blocks
from src.config import CAPACITY

# Judge module to manage the judge model during training and MemRecall procedure
class JudgeModule(pl.LightningModule):

    def __init__(self, config):
        super(JudgeModule, self).__init__()
        self.config = config
        self.hparams = deepcopy(config)
        if hasattr(self.hparams, 'gpus'):
            del self.hparams.gpus
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.judge = Judge.from_pretrained(config.model_name)

    def on_save_checkpoint(self, checkpoint): 
        # to fix the bug of pytorch-lightning 6.0.0, will remove for future versions
        checkpoint['epoch'] += 1
        checkpoint['global_step'] += 1
        # print('\nSaved judge!')

    def validation_step(self, batch, batch_idx):
        pass

    def validation_end(self, outputs):
        return {'val_loss': -self.current_epoch}

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            dataset=range(8),
            shuffle=True,
            batch_size=1,
            num_workers=0
        )

    def forward(self, x):
        pass

    def on_epoch_start(self):
        os.makedirs(self.config.tmp_dir, exist_ok=True)
        # self.device = next(self.judge.parameters()).device
        self.device = torch.device("cpu")
        device_name = str(self.device).replace(':', '_')
        self._file = open(os.path.join(self.config.tmp_dir, 'estimations_{}.txt'.format(device_name)), 'w')

    def on_epoch_end(self):
        self._file.close()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.judge.parameters(),
            lr=self.config.lr_judge,
            weight_decay=self.config.weight_decay_judge
            )

        scheduler = WarmupLinearLR(optimizer, self.config.step_size)

        return [optimizer], [scheduler]

    def set_dataset(self, dataset, mode='train'):
        if mode == 'train':
            self.train_dataset = dataset
        elif mode == 'val':
            self.val_dataset = dataset
        elif mode == 'test':
            self.test_dataset = dataset
        else:
            raise ValueError('No such dataset')

    @pl.data_loader
    def train_dataloader(self):
        # when using multi-node (ddp) we need to add the  datasampler
        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size_judge_per_gpu,
            shuffle=True,
            num_workers=0,
            collate_fn=buffer_collate
        )
        logging.info('train_dataset reloaded in Judge.')

        return loader

    def _write_estimation(self, buf, relevance_blk):
        for i, blk in enumerate(buf):
            self._file.write(f'{blk.pos} {relevance_blk[i].item()}\n')

    def training_step(self, bufs, batch_idx):
        # Make inputs for judge
        inputs = torch.zeros(4, len(bufs), CAPACITY, dtype=torch.long, device=self.device)

        for i, buf in enumerate(bufs):
            buf.export(out=(inputs[0, i], inputs[1, i], inputs[2, i]))

        # Train the judge after labeling
        for i, buf in enumerate(bufs):
            buf.export_relevance(device=self.device, out=inputs[3, i])

        # Label the relevance by the current judge
        loss_judge, logits = self.judge(*inputs[:3], labels=inputs[3])
        for i, buf in enumerate(bufs):
            self._write_estimation(buf, _score_blocks(buf, torch.sigmoid(logits[i])))

        tensorboard_logs = {'loss': loss_judge}
        
        return {'loss': loss_judge, 'log': tensorboard_logs}

    @staticmethod
    def add_specific_args(parser):
        parser.add_argument('--lr_judge', type=float, default=4e-5, help='learning rate of judge')
        parser.add_argument('--weight_decay_judge', type=float, default=0.01, help='weight decay of judge')
        parser.add_argument('--batch_size_judge_per_gpu', type=int, default=2, help='gradient batch_size')

