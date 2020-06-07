class Vocab:
    
    def __init__(self, sents):
        
        # Special tokens. Every vocabulary must include those special tokens
        # whether use or do not
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        self.w2i = self.build(sents)
        self.i2w = {idx: word for word, idx in self.w2i.items()}

    def __len__(self):
        """
        Length of the vocabulary.
        """
        return len(self.w2i)
    
    def build(self, sents):
        """
        Generate the vocabulary from given list of sentences.
        additional_tokens: list of additional tokens to
        include in vocabulary.
        """
        # Special characters
        w2i = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        num_words = len(w2i)        
        for sent in sents:
            for token in sent:
                if token in w2i:
                    continue
                else:
                    w2i[token] = num_words
                    num_words += 1

        return w2i
    
    def word2index(self, sents):
        """
        Convert list of tokenized sentences into the list of indices. 
        """
        
        return [[self.w2i[token] if token in self.w2i else self.unk_idx for token in sent] for sent in sents]
    
    
    def index2word(self, sents):
        """
        Convert list of token id's into the list of sentences.
        """
        return [[self.i2w[token] for token in sent] for sent in sents]