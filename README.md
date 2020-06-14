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

# Improvements

# References