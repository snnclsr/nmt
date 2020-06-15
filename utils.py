import math
from typing import List, Tuple

from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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



def save_attention(src, pred, attention_weights, save_path="attention_map.png"):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    attention_weights = np.array(attention_weights).transpose()
    ax.matshow(attention_weights, cmap='bone')
    ax.set_xticklabels([''] + pred, rotation=90)
    ax.set_yticklabels([''] + src)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(save_path)
    # plt.show()


def generate_attention_map(model, vocabs, test_src, test_tgt):
    """
    This function is actually the decoding step of the Seq2Seq model.
    We only use the attention weights.
    TODO: add this functionality to the beam search as a parameter
    since it already does the decoding step in itself (duplicate code for now).
    """

    examples = list(zip(test_src, test_tgt))

    device = "cpu"
    examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)

    src_sents, tgt_sents = zip(*examples)
    source_lengths = torch.tensor([len(s) for s in src_sents])
    src_tensor = to_tensor(vocabs.src, src_sents, device=device) # (max_seq_len, bs)
    tgt_tensor = to_tensor(vocabs.tgt, tgt_sents, device=device) # (max_seq_len, bs)

    enc_hiddens, dec_init_state = model.encoder(src_tensor, source_lengths)
    enc_hiddens_proj = model.decoder.attn_projection(enc_hiddens)

    Y = model.decoder.embedding(tgt_tensor)

    dec_state = dec_init_state
    batch_size = enc_hiddens.size(0)
    o_prev = torch.zeros(batch_size, model.decoder.hidden_size, device=device)
    enc_masks = generate_sent_masks(enc_hiddens, source_lengths, device=device)

    a_ts = []
    combined_outputs = []
    for y_t in torch.split(Y, 1, dim=0):
        y_t = y_t.squeeze(dim=0)
        ybar_t = torch.cat((y_t, o_prev), dim=1)
        dec_state, o_t, a_t = model.decoder.step(ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
        a_ts.append(a_t)
        combined_outputs.append(o_t)
        o_prev = o_t

    return a_ts


def beam_search(model, test_data, beam_size, max_decoding_time_step):
  
    model.eval()
    hypotheses = []

    with torch.no_grad():
        for sent in tqdm(test_data):
            hyp = model.beam_search(sent, beam_size, max_decoding_time_step)
            hypotheses.append(hyp)

    return hypotheses


def compute_corpus_level_bleu_score(references, hypotheses) -> float:
    """ 
    Given decoding results and reference sentences, compute corpus-level BLEU score.
    """
    if references[0][0] == '<sos>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score

