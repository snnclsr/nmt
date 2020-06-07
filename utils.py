import math
from typing import List, Tuple

import numpy as np

import torch

def read_text(filename: str) -> List[str]:
    
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read().split('\n')[:-1]
        
    return data

def add_start_end_tokens(dataset: List[Tuple[str, str]]) -> Tuple[List[List[str]], List[List[str]]]:

    src_data = []
    tgt_data = []
    for src, tgt in dataset:
        src_sent = src.strip().split()
        tgt_sent = ["<sos>"] + tgt.strip().split() + ["<eos>"]
        src_data.append(src_sent)
        tgt_data.append(tgt_sent)

    return src_data, tgt_data     


def pad_sents(sents, pad_idx):
    """
    Pad the sentences with respect to max length sentence.
    """
    max_len = max([len(sent) for sent in sents])
    padded_sents = []
    for sent in sents:
        if len(sent) < max_len:
            sent = sent + [pad_idx] * (max_len - len(sent))
        
        padded_sents.append(sent)
            
    return padded_sents


def generate_sent_masks(sents, lengths, device=torch.device("cpu")):
    """
    Generate the padding masking for given sents from lenghts. 
    Assumes lengths are sorted by descending order.
    """
    max_len = lengths[0]
    bs = sents.shape[0]
    mask = ~(torch.arange(max_len).expand(bs, max_len) < lengths.unsqueeze(1))
    return mask.bool().to(device)

def to_tensor(vocab, sents, device=torch.device("cpu")):
    
    sent_indices = vocab.word2index(sents)
    padded_sents = pad_sents(sent_indices, vocab.pad_idx)
    sent_tensor = torch.tensor(padded_sents, dtype=torch.long, device=device)
    return torch.t(sent_tensor) # (max_seq_len, batch_size)


def batch_iter(data, batch_size, shuffle=False):
    
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))
    if shuffle:
        np.random.shuffle(index_array)
        
    for i in range(batch_num):
        indices = index_array[i * batch_size: (i+1) * batch_size]
        examples = [data[idx] for idx in indices]
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]
        
        yield src_sents, tgt_sents