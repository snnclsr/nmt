import argparse
import pickle

from vocab import Vocab, Vocabularies
from models import Seq2Seq
from utils import read_text, add_start_end_tokens, beam_search, compute_corpus_level_bleu_score

import numpy as np
import torch


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