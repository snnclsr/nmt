import argparse
import pickle

from vocab import Vocab
from models import Seq2Seq
from utils import to_tensor, generate_sent_masks, generate_attention_map, add_start_end_tokens, read_text
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from train import Vocabularies

from tqdm import tqdm

import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def beam_search(model, test_data, beam_size, max_decoding_time_step):
  
    model.eval()
    hypotheses = []

    with torch.no_grad():
        for sent in tqdm(test_data):
            hyp = model.beam_search(sent, beam_size, max_decoding_time_step)
            hypotheses.append(hyp)

    return hypotheses


def compute_corpus_level_bleu_score(references, hypotheses) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<sos>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score



def show_attention(src, pred, attention_weights):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    attention_weights = np.array(attention_weights).transpose()
    cax = ax.matshow(attention_weights, cmap='bone')
    ax.set_xticklabels([''] + pred, rotation=90)
    ax.set_yticklabels([''] + src)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig("attention_map.png")
    plt.show()


def main():
    
    arg_parser = argparse.ArgumentParser(description="Neural Machine Translation Testing")
    arg_parser.add_argument("--model_file", required=True, help="Model File")
    arg_parser.add_argument("--valid_data", required=True, nargs="+", help="Validation_data")

    args = arg_parser.parse_args()
    args = vars(args)
    print(args)
    model = Seq2Seq.load(args["model_file"])
    print(model)
    model.device = "cpu"

    tr_dev_dataset_fn, en_dev_dataset_fn = args["valid_data"]
    tr_valid_data = read_text(tr_dev_dataset_fn)
    en_valid_data = read_text(en_dev_dataset_fn)

    valid_data = list(zip(tr_valid_data, en_valid_data))

    src_valid, tgt_valid = add_start_end_tokens(valid_data)

    hypotheses = beam_search(model, src_valid, beam_size=3, max_decoding_time_step=70)
    top_hypotheses = [hyps[0] for hyps in hypotheses]
    bleu_score = compute_corpus_level_bleu_score(tgt_valid, top_hypotheses)
    print('Corpus BLEU: {}'.format(bleu_score * 100))

if __name__ == "__main__":
    main()