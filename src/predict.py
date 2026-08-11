# Code for inference adapted from:
# Ding et al. (2020), "CogLTX: Applying BERT to Long Texts"
# https://github.com/Sleepychord/CogLTX
# License: MIT

import torch

from tqdm import tqdm
from pathlib import Path

from external.cogltx.data_helper import SimpleListDataset,find_latest_checkpoint 
from external.cogltx.judge_module import JudgeModule
from external.cogltx.reasoner_module import ReasonerModule
from external.cogltx.memrecall import mem_recall

def prediction(config, test_dataset_filename):
    # device = f'cuda:{config.gpus[0]}'
    device = 'cpu'

    judge_module = JudgeModule.load_from_checkpoint(find_latest_checkpoint(Path(config.save_dir) / 'judge' / f'version_{config.version}' / 'checkpoints')).to(device).eval()
    reasoner_module = ReasonerModule.load_from_checkpoint(find_latest_checkpoint(Path(config.save_dir) / 'reasoner' / f'version_{config.version}' / 'checkpoints')).to(device).eval()

    test_dataset = SimpleListDataset(test_dataset_filename)
    with torch.no_grad():
        for qbuf, dbuf in tqdm(test_dataset):
            
            # 4 baris berikut untuk demo
            # ticker = qbuf[0].filename.split('_MDA_')[0]
            # year = int(qbuf[0].filename.split('_MDA_')[1].split('.txt')[0])
            # if ticker == 'SHID' and year == 2021:
            # print(f"Processing file: SHID_MDA_2021.txt")

            buf, relevance_score = mem_recall(judge_module.judge, qbuf, dbuf, times=config.times, device=device, batch_size_inference=config.batch_size_inference)
            
            inputs = [t.unsqueeze(0) for t in buf.export(device=device)]

            # 5 baris berikut untuk demo
            # input_ids = inputs[0][0]
            # text = reasoner_module.tokenizer.decode(input_ids, skip_special_tokens=True)
            # print()
            # print(f"Text selected by MemRecall:")
            # print(text)

            output = reasoner_module.reasoner(*inputs)
            
            yield qbuf, dbuf, buf, relevance_score, inputs[0][0], output