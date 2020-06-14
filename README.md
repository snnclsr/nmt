# Neural Machine Translation

This repository implements a Turkish to English Neural Machine Translation system using Seq2Seq + Global Attention model.

# Examples

# Dataset

The dataset for this project is taken from [here](http://opus.nlpl.eu/). I have used the [Tatoeba](http://opus.nlpl.eu/Tatoeba-v20190709.php) corpus. I have deleted some of the duplicates found in the data. But there are some examples left since there can be multiple translations for the same sentence.

## Tokenization

* For tokenizing the Turkish sentences, I've used the nltk's [RegexpTokenizer](http://www.nltk.org/api/nltk.tokenize.html?highlight=regexp#nltk.tokenize.regexp.RegexpTokenizer). 

```python
    puncts_except_apostrophe = '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~'
    TOKENIZE_PATTERN = fr"[{puncts_except_apostrophe}]|\w+|['\w]+"
    regex_tokenizer = RegexpTokenizer(pattern=TOKENIZE_PATTERN)
    text = "Titanic 15 Nisan pazartesi saat 02:20'de battı."
    tokenized_text = regex_tokenizer.tokenize(text)
    print(" ".join(tokenized_text))
    # Output: Titanic 15 Nisan pazartesi saat 02 : 20 'de battı .
    # This splitting property on "02 : 20" is different from the English tokenizer.
    # We could handle those situations. But I wanted to keep it simple and see if 
    # the attention distribution on those words aligns with the English tokens.
    # There are similar cases mostly on dates as well like in this example: 02/09/2019
```

* For tokenizing the English sentences, I've used the spacy's English model.

```python
    en_nlp = spacy.load('en_core_web_sm')
    text = "The Titanic sank at 02:20 on Monday, April 15th."
    tokenized_text = en_nlp.tokenizer(text)
    print(" ".join([tok.text for tok in tokenized_text]))
    # Output: The Titanic sank at 02:20 on Monday , April 15th .
```

## Format

Turkish and English sentences are expected to be in two different files.

```
file: train.tr
tr_sent_1
tr_sent_2
tr_sent_3
...

file: train.en
en_sent_1
en_sent_2
en_sent_3
...
```


# Train

**Please run `python train.py -h` for the full list of arguments.**

```
Sample usage:

python train.py --train_data train.tr train.en --valid_data valid.tr valid.en --n_epochs 30 --batch_size 32 --embedding_dim 256 --hidden_size 256 --num_layers 2 --bidirectional --dropout_p 0.3 --device cuda
```

```
usage: train.py [-h] --train_data TRAIN_DATA [TRAIN_DATA ...] --valid_data
                VALID_DATA [VALID_DATA ...] [--n_epochs N_EPOCHS]
                [--batch_size BATCH_SIZE] [--embedding_dim EMBEDDING_DIM]
                [--hidden_size HIDDEN_SIZE] [--num_layers NUM_LAYERS]
                [--bidirectional] [--dropout_p DROPOUT_P]
                [--initial_lr INITIAL_LR] [--uniform_init UNIFORM_INIT]
                [--clip_grad CLIP_GRAD] [--lr_decay LR_DECAY]
                [--patience PATIENCE] [--max_trial MAX_TRIAL]
                [--device DEVICE] [--model_name MODEL_NAME]

Neural Machine Translation Training

optional arguments:
  -h, --help            show this help message and exit
  --train_data TRAIN_DATA [TRAIN_DATA ...]
                        Parallel training data
  --valid_data VALID_DATA [VALID_DATA ...]
                        Parallel validation data
  --n_epochs N_EPOCHS
  --batch_size BATCH_SIZE
  --embedding_dim EMBEDDING_DIM
                        Embedding dimension for the word embeddings
  --hidden_size HIDDEN_SIZE
                        RNN hidden dimension
  --num_layers NUM_LAYERS
                        Number of RNN Layers
  --bidirectional       Whether or not bidirectional RNNs
  --dropout_p DROPOUT_P
                        Dropout probability for word embeddings and Decoder
                        networks output
  --initial_lr INITIAL_LR
                        Initial learning rate for the optimizer
  --uniform_init UNIFORM_INIT
                        Uniformly initialization of the model's parameter
  --clip_grad CLIP_GRAD
                        Gradient clipping value to be applied to the model
  --lr_decay LR_DECAY   Learning rate decay if the validation metric doesn't
                        improve
  --patience PATIENCE   Learning rate decay patience
  --max_trial MAX_TRIAL
                        Maximum number of trials for early stopping
  --device DEVICE       Device to run the model
  --model_name MODEL_NAME
                        Model name
```

# Improvements

* Using subword units (for both Turkish and English)
* Different attention mechanisms (learning different parameters for the attention)

# References