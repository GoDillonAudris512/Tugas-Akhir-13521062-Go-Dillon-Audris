# Code for training adapted from:
# Ding et al. (2020), "CogLTX: Applying BERT to Long Texts"
# https://github.com/Sleepychord/CogLTX
# License: MIT

import logging

from copy import copy
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from transformers import AutoTokenizer
from pathlib import Path

from external.cogltx.data_helper import SimpleListDataset, BlkPosInterface, find_latest_checkpoint
from external.cogltx.initialize_relevance import init_relevance
from external.cogltx.judge_module import JudgeModule
from external.cogltx.reasoner_module import ReasonerModule
from external.cogltx.buffer import Buffer

def train_model(config, train_dataset_filename):
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Create a new trainer
    def _create_new_trainer(num_epochs, logger):
        return Trainer(
            max_epochs=num_epochs,
            # gpus=config.gpus,
            gpus=None,
            default_save_path=config.save_dir,
            logger=logger,
            weights_summary=None,
            early_stop_callback=False,
            check_val_every_n_epoch=1,
        )

    # Define transformation for query buffer for classification task
    def classification_conditional_transforms(qbuf, dbuf):
        assert len(qbuf) == 1 

        new_qbuf = Buffer()
        new_qblk = copy(qbuf[0])
        new_qblk.ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(new_qblk.label_name))
        new_qbuf.blocks.append(new_qblk)

        return new_qbuf, dbuf

    logger_judge = TensorBoardLogger(config.log_dir, name='judge', version=config.version)
    logger_reasoner = TensorBoardLogger(config.log_dir, name='reasoner', version=config.version)

    train_dataset = SimpleListDataset(train_dataset_filename)
    train_interface = BlkPosInterface(train_dataset)

    # Initialize relevance
    if config.init_relevance != '':
        init_relevance(train_dataset, method=config.init_relevance, conditional_transforms=[classification_conditional_transforms])

    # Initialize modules
    judge = JudgeModule(config)
    reasoner = ReasonerModule(config)

    # Training loop
    min_epoch = min(
        find_latest_checkpoint(Path(config.save_dir) / 'judge' / f'version_{config.version}' / 'checkpoints', epoch=True),
        find_latest_checkpoint(Path(config.save_dir) / 'reasoner' / f'version_{config.version}' / 'checkpoints', epoch=True)
    ) + 1

    logging.info(f'Continue training at epoch {min_epoch}...')

    for epoch in range(min_epoch, config.num_epochs):
        # Train judge
        judge_dataset = train_interface.build_random_buffer(num_samples=config.num_samples)
        judge.set_dataset(judge_dataset)

        trainer = _create_new_trainer(epoch + 1, logger_judge)
        trainer.fit(judge)

        # Train reasoner
        train_interface.collect_estimations_from_dir(config.tmp_dir)
        reasoner_dataset = train_interface.build_promising_buffer(num_samples=config.num_samples)
        reasoner.set_dataset(reasoner_dataset)

        trainer = _create_new_trainer(epoch + 1, logger_reasoner)
        trainer.fit(reasoner)

        # Relevance labels update
        if config.latent and epoch > 1:
            train_interface.apply_changes_from_dir(config.tmp_dir)