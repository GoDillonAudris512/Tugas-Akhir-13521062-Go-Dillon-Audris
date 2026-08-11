# Original code from:
# Ding et al. (2020), "CogLTX: Applying BERT to Long Texts"
# https://github.com/Sleepychord/CogLTX
# License: MIT
#
# Change in this file:
# - Changed file name from memreplay.py to memrecall.py 
# - Added comments for clarity
# - Minor formatting (no functional changes)
# - Changed the word choice mem-replay to mem-recall (according to paper)
# - Changed the word choice introspector to judge (according to paper)

import torch
import torch.nn.functional as F

from external.cogltx.buffer import Buffer
from src.config import CAPACITY

# Compute relevance scores for blocks in the buffer by averaging token-level relevance scores
def _score_blocks(qbuf, relevance_token):
    ends = qbuf.block_ends()
    relevance_blk = torch.ones(len(ends), device='cpu')

    for i in range(len(ends)): 
        if qbuf[i].blk_type > 0: # query
            relevance_blk[i] = (relevance_token[ends[i-1]:ends[i]]).mean()

    return relevance_blk

# Apply positional smoothing to relevance scores
def positional_smoothing(buf, relevance_blk, factor_forward=0.1, factor_backward=0.3):
    ret = torch.zeros_like(relevance_blk)
    for i, blk in enumerate(buf):
        rest = 1.   

        if i > 0 and buf[i-1].pos == blk.pos - 1:
            rest -= factor_forward
            ret[i] += relevance_blk[i-1] * factor_forward

        if i < len(buf) - 1 and buf[i+1].pos == blk.pos + 1:
            rest -= factor_backward
            ret[i] += relevance_blk[i+1] * factor_backward

        ret[i] += relevance_blk[i] * rest
        ret[i] = max(ret[i], relevance_blk[i])

    return ret

# MemRecall procedure
def mem_recall(judge, qbuf, dbuf, device, times='3,5', batch_size_inference=16):

    # print(f"The text is divided into {len(dbuf)} blocks")

    '''
        times: increased number of blocks each recall.
    '''
    times = [int(x) for x in times.split(',')]
    inputs = torch.zeros(3, batch_size_inference, CAPACITY, dtype=torch.long, device=device)

    B_set = [] # the poses of B blks in qbuf
    for k, inc in enumerate(times):
        # print(f"\n// MEM-RECALL STEP {k+1} //")
        num_to_keep = len(qbuf) + inc

        # stage one: continuous (compute relevance scores for blocks)
        estimations = torch.zeros(len(dbuf), device='cpu')
        bufs, t = qbuf.fill(dbuf), 0
        for i in range((len(bufs) - 1) // batch_size_inference + 1):
            l, r = batch_size_inference * i, min(len(bufs), batch_size_inference * (i + 1))
            for j, buf in enumerate(bufs[l:r]):
                buf.export(out=(inputs[0, j], inputs[1, j], inputs[2, j]))

            logits = judge(*inputs[:,:r-l]).sigmoid_()
            for j, buf in enumerate(bufs[l:r]):
                estimation = _score_blocks(buf, logits[j])[len(qbuf):]
                estimations[t: t + len(estimation)] = estimation
                t += len(estimation)

        assert t == len(dbuf)

        # print(f"\n// RETRIEVAL COMPETITION //")
        # print(f"After retrieval competition, here are the relevance scores for each block in the text:")
        # print(estimations)
        # print()

        # estimations = positional_smoothing(dbuf, estimations)
        # fill the buffer up
        indices = estimations.argsort(descending=True)
        qbuf_size = qbuf.calc_size()

        # Variable for demo
        # total_blocks_retrieved = 0
        # idx_list = []

        for idx in indices:
            if qbuf_size + len(dbuf[idx]) > CAPACITY:
                # print(f"Buffer is full. Cannot add block {idx} with size {len(dbuf[idx])}.")
                break

            if dbuf[idx] in B_set:
                continue

            qbuf_size += len(dbuf[idx])
            qbuf.insert(dbuf[idx])

            # total_blocks_retrieved += 1
            # print(f"Block {idx} added to buffer with relevance score {estimations[idx]:.4f} and size {len(dbuf[idx])}.")
            # idx_list.append(idx)
        # print(f"Total blocks added in this round: {total_blocks_retrieved}")

        # print(f"\nBuffer after retrieval competition:")
        # idx_list.sort()
        # print(f"Buffer block arrangement: {[t.item() for t in idx_list]}")
        # print(qbuf)


        # keep only num_to_keep blks
        qbuf.export(out=(inputs[0, 0], inputs[1, 0], inputs[2, 0]))
        relevance_token = torch.sigmoid(judge(*inputs[:, :1]).view(-1))
        relevance_blk = _score_blocks(qbuf, relevance_token)

        # print(f"// REHEARSAL //")
        # print(f"After rehearsal, here are the relevance scores for each block in the buffer:")
        # print(f"Buffer block arrangement: {[t.item() for t in idx_list]}")
        # print(relevance_blk[1:])

        # print(f"\n// DECAY //")
        keeped_indices = relevance_blk.argsort(descending=True)
        if len(keeped_indices) > num_to_keep and k < len(times) - 1:
            keeped_indices = keeped_indices[:num_to_keep]
        else:   
            # print(f"All blocks are kept in the buffer")
            # print(f"\nBuffer after decay:")
            # print(qbuf)
            return qbuf, relevance_blk

        # manually filtering
        # print(f"Keeping the top {inc} blocks. Remaining blocks will be removed.")

        filtered_qbuf, filtered_relevance_blk = Buffer(), []
        for i, blk in enumerate(qbuf):
            if i in keeped_indices:
                # if i > 0: print(f"Block {i-1} kept in buffer")
                filtered_qbuf.blocks.append(blk)
                filtered_relevance_blk.append(relevance_blk[i])

        qbuf = filtered_qbuf

        # print(f"\nBuffer after decay:")
        # print(qbuf)

        # record the blocks already in the qbuf
        B_set = [blk for blk in qbuf if blk.blk_type == 1]

    return filtered_qbuf, torch.tensor(filtered_relevance_blk)

