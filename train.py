import argparse
import logging
import time
import datetime
import random
import pickle
import json
from pathlib import Path
from collections import namedtuple
from typing import List, Tuple

from utils import read_text, add_start_end_tokens, batch_iter
from vocab import Vocab
from models import Seq2Seq

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

Vocabularies = namedtuple("Vocabularies", "src tgt")
EXPERIMENTS_DIR = Path("experiments")


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


def evaluate_ppl(model, dev_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += float(loss)
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)
        

    if was_training:
        model.train()

    return ppl


def train(model, optimizer, train_data, valid_data, args):

    batch_size = args["batch_size"]

    data_len = len(train_data)
    epoch_iteration = data_len // batch_size

    n_epochs = args["n_epochs"]
    clip_grad = args["clip_grad"]
    model_save_path = args["model_save_path"]
    lr_decay = args["lr_decay"]
    num_trial = args["num_trial"]
    epoch_patience = args["patience"]
    max_trial = args["max_trial"]
    device = args["device"]
    hist_valid_scores = []
    epoch_counter = 0

    while epoch_counter < n_epochs:

        model.train()
        total_train_loss = 0.0
        epoch_start_time = time.time()
        total_train_iteration = len(train_data) // batch_size
        for src_sents, tgt_sents in tqdm(batch_iter(train_data, batch_size=batch_size, shuffle=True), total=total_train_iteration):
            
            optimizer.zero_grad()
            loss = (-model(src_sents, tgt_sents).sum()) / batch_size
            loss.backward()
            total_train_loss += float(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        
        epoch_time = str(datetime.timedelta(seconds=round(time.time() - epoch_start_time)))
        logger.info("Epoch {} done in: {}".format(epoch_counter+1, epoch_time))

        valid_start_time = time.time()
        model.eval()
        with torch.no_grad():

            # compute dev. ppl and bleu
            dev_ppl = evaluate_ppl(model, valid_data, batch_size=128)
            valid_metric = -dev_ppl

            is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
            hist_valid_scores.append(valid_metric)
            
            if is_better:
                patience = 0
                logger.info("Saving the model...")
                model.save(model_save_path)
                torch.save(optimizer.state_dict(), model_save_path + '.optim')

            elif patience < epoch_patience:
                patience += 1

                if patience == epoch_patience:
                    num_trial += 1

                    if num_trial == max_trial:
                        print("Early Stopping hit.")
                        break

                    # Decaying the learning rate.
                    lr = optimizer.param_groups[0]['lr'] * lr_decay
                    logger.info("Loading the previous best model. Decayed the learning rate to: {}".format(lr))
                    params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                    model.load_state_dict(params["state_dict"])
                    model.to(device)

                    optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr

                    patience = 0


        train_loss = total_train_loss / total_train_iteration
        valid_ppl = hist_valid_scores[-1]
        logger.info("Epoch: {:02d}/{} Loss: {:.4f} Valid_Ppl: {:.4f}".format(epoch_counter+1, 
                                                                        n_epochs, 
                                                                        train_loss, 
                                                                        valid_ppl))

        valid_end_time = str(datetime.timedelta(seconds=round(time.time() - valid_start_time)))
        logger.info("Validation done in: {}\n".format(valid_end_time))
        epoch_counter += 1


def main():
    
    arg_parser = argparse.ArgumentParser(description="Neural Machine Translation Training")
    arg_parser.add_argument("--train_data", required=True, nargs="+", 
                            help="Parallel training data")
    arg_parser.add_argument("--valid_data", required=True, nargs="+",
                            help="Parallel validation data")
    arg_parser.add_argument("--n_epochs", type=int, default=30, help="")
    arg_parser.add_argument("--batch_size", type=int, default=32, help="")
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
    arg_parser.add_argument("--initial_lr", type=float, default=0.001, 
                            help="")
    arg_parser.add_argument("--uniform_init", type=float, default=0.0, 
                            help="")
    arg_parser.add_argument("--clip_grad", type=float, default=5.0, 
                            help="Gradient clipping value to be applied to the model")
    arg_parser.add_argument("--lr_decay", type=float, default=0.5, help="")
    arg_parser.add_argument("--patience", type=int, default=5, help="")
    arg_parser.add_argument("--num_trial", type=int, default=5, help="")
    arg_parser.add_argument("--max_trial", type=int, default=5, help="")
    arg_parser.add_argument("--device", type=str, default="cpu", 
                            help="Device to run the model")
    arg_parser.add_argument("--model_name", type=str, default="model.bin", help="")
    args = arg_parser.parse_args()
    args = vars(args)
    print(args)

    device = "cuda" if args["device"] == "cuda" else "cpu"
    if not torch.cuda.is_available() and args["device"] == "cuda":
        logger.info("Device is specified as cuda. But there is no cuda device available in your system.")
        exit(0)


    if not EXPERIMENTS_DIR.exists():
        EXPERIMENTS_DIR.mkdir()

    current_experiment_name = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    current_experiment_dir = Path(EXPERIMENTS_DIR / current_experiment_name)
    # Create a new experiment directory based on timestamp.
    current_experiment_dir.mkdir()

    # Save the command line arguments to the file.    
    with open(current_experiment_dir / "params.json", "w") as f:
        json.dump(args, f, indent=4)

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
    
    with open(current_experiment_dir / "vocabs.pkl", "wb") as f:
        pickle.dump(vocabs, f)

    logger.info("Total words in the source language: {}".format(len(src_vocab)))
    logger.info("Total words in the target language: {}".format(len(tgt_vocab)))

    model = Seq2Seq(vocabs=vocabs, embedding_dim=args["embedding_dim"], hidden_size=args["hidden_size"], 
                    num_layers=args["num_layers"], bidirectional=args["bidirectional"], 
                    dropout_p=args["dropout_p"], device=device)
    model.to(device)
    print(model)

    if args["uniform_init"] > 0:
        for p in model.parameters():
            p.data.uniform_(-args["uniform_init"], args["uniform_init"])

    optimizer = optim.Adam(model.parameters(), lr=args["initial_lr"])
    # train_data = train_data[:100]
    # valid_data = valid_data[:100]
 
    args["model_save_path"] = str(current_experiment_dir / args["model_name"])
    train(model, optimizer, train_data, valid_data, args)


if __name__ == "__main__":
    main()