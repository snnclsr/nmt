import argparse
import logging
import random
from collections import namedtuple
from typing import List, Tuple

import torch

from utils import read_text, add_start_end_tokens
from vocab import Vocab
from models import Seq2Seq

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

Vocabularies = namedtuple("Vocabularies", "src tgt")


def print_random_samples(data: List[Tuple[str, str]], n: int=5):
    """
    Print the randomly selected samples from the given dataset.
    """
    indices = random.sample(range(0, len(data)), k=n)
    for idx in indices:
        tr_sent, en_sent = data[idx]
        print("TR: ", tr_sent)
        print("EN: ", en_sent)
        print("="*50)


def main():
    
    arg_parser = argparse.ArgumentParser(description="Neural Machine Translation Training")
    arg_parser.add_argument("--train_data", required=True, nargs="+", 
                            help="Parallel training data")
    arg_parser.add_argument("--valid_data", required=True, nargs="+",
                            help="Parallel validation data")
    arg_parser.add_argument("--embedding_dim", type=int, default=64,
                            help="Embedding dimension for the word embeddings")
    arg_parser.add_argument("--hidden_size", type=int, default=64,
                            help="")
    arg_parser.add_argument("--num_layers", type=int, default=1, 
                            help="")
    arg_parser.add_argument("--bidirectional", action="store_true", 
                            help="")
    arg_parser.add_argument("--dropout_p", type=float, default=0.1, 
                            help="")
    arg_parser.add_argument("--device", type=str, default="cpu", 
                            help="Device to run the model")
    args = arg_parser.parse_args()
    args = vars(args)
    print(args)

    device = "cuda" if args["device"] == "cuda" else "cpu"
    if not torch.cuda.is_available() and args["device"] == "cuda":
        logger.info("Device is specified as cuda. But there is no cuda device available in your system.")
        exit(0)


    tr_train_dataset_fn, en_train_dataset_fn = args["train_data"]
    tr_dev_dataset_fn, en_dev_dataset_fn = args["valid_data"]

    tr_train_data = read_text(tr_train_dataset_fn)
    en_train_data = read_text(en_train_dataset_fn)

    tr_valid_data = read_text(tr_dev_dataset_fn)
    en_valid_data = read_text(en_dev_dataset_fn)

    logger.info("Total train sentences: {}".format(len(tr_train_data)))
    logger.info("Total valid sentences: {}".format(len(tr_valid_data)))

    train_data = list(zip(tr_train_data, en_train_data))
    valid_data = list(zip(tr_valid_data, en_valid_data))

    logger.info("Random samples from training data")
    print_random_samples(train_data, n=3)
    logger.info("Random samples from validation data")
    print_random_samples(valid_data, n=3)

    src_train, tgt_train = add_start_end_tokens(train_data)
    src_valid, tgt_valid = add_start_end_tokens(valid_data)

    train_data = list(zip(src_train, tgt_train))
    valid_data = list(zip(src_valid, tgt_valid))

    src_vocab = Vocab(src_train)
    tgt_vocab = Vocab(tgt_train)

    vocabs = Vocabularies(src_vocab, tgt_vocab)
    
    logger.info("Total words in the source language: {}".format(len(src_vocab)))
    logger.info("Total words in the target language: {}".format(len(tgt_vocab)))
    
    model = Seq2Seq(vocabs=vocabs, embedding_dim=args["embedding_dim"], hidden_size=args["hidden_size"], 
                    num_layers=args["num_layers"], bidirectional=args["bidirectional"], 
                    dropout_p=args["dropout_p"], device=device)
    model.to(device)
    print(model)


if __name__ == "__main__":
    main()