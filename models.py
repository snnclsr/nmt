from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import to_tensor, generate_sent_masks

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class Encoder(nn.Module):
    
    def __init__(self, vocab, embedding_dim, hidden_size, num_layers=1, bidirectional=True):
        super(Encoder, self).__init__()
        
        self.vocab = vocab
        self.embedding = nn.Embedding(num_embeddings=len(vocab), 
                                      embedding_dim=embedding_dim,
                                      padding_idx=vocab.pad_idx)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, 
                            bidirectional=bidirectional)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        proj_hidden = hidden_size
        if bidirectional:
            proj_hidden = hidden_size * 2

        self.h_projection = nn.Linear(proj_hidden, hidden_size, bias=False)
        self.c_projection = nn.Linear(proj_hidden, hidden_size, bias=False)
    
    def forward(self, src_sents, src_lengths):
        
        X = self.embedding(src_sents) # (max_seq_len, bs, embedding_dim)        
        X = pack_padded_sequence(X, src_lengths)
        hidden_outs, (h_n, c_n) = self.lstm(X)
        hidden_outs, _ = pad_packed_sequence(hidden_outs)
        hidden_outs.transpose_(0, 1) # (max_seq_len, bs, hs*2) -> (bs, max_seq_len, hs*2)
        
        if self.bidirectional:
            h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
            c_n = torch.cat((c_n[-2], c_n[-1]), dim=1)
                
        initial_decoder_hidden = self.h_projection(h_n)
        initial_decoder_cell = self.c_projection(c_n)
        
        return hidden_outs, (initial_decoder_hidden, initial_decoder_cell)

        
class Decoder(nn.Module):
    
    def __init__(self, vocab, embedding_dim, hidden_size, dropout_rate=0.2, device=torch.device("cpu")):
        super(Decoder, self).__init__()
        
        self.vocab = vocab
        self.embedding = nn.Embedding(num_embeddings=len(vocab), 
                            embedding_dim=embedding_dim, 
                            padding_idx=vocab.pad_idx)
        
        self.hidden_size = hidden_size
        self.attn_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.lstm_cell = nn.LSTMCell(embedding_dim + hidden_size, hidden_size)
        self.combined_output_projection = nn.Linear(3 * hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.vocab_projection = nn.Linear(hidden_size, len(vocab), bias=False)
        self.device = device
    
    def forward(self, enc_hiddens, enc_masks, initial_state, tgt_sents):
        """
        enc_hiddens: encoder's hidden states for all timesteps. (bs, max_seq_len, 2 * h_s)
        enc_masks: mask for the source sentences (bs, src_len)
        initial_state: initial state of the decoder (e.g. output (last) state of the encoder) (bs, h_s)
        tgt_sents: target sentences (tgt_len, b)
        """
        
        # Target sents includes <eos> at the end of sentence. 
        tgt_sents = tgt_sents[:-1]
        dec_state = initial_state
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)
        
        combined_outputs = []
        enc_hidden_projections = self.attn_projection(enc_hiddens) # (bs, max_seq_len, hs)        
        Y = self.embedding(tgt_sents) # (max_seq_len, bs, embedding_dim)

        # Taking every single word embedding for timesteps t.
        for y_t in torch.split(Y, 1):
            y_t = y_t.squeeze(dim=0) # (1, bs, embedding_dim) -> (bs, embedding_dim)
            y_t = torch.cat((y_t, o_prev), dim=1) # (bs, embedding_dim + hs)
            dec_state, o_t, _ = self.step(y_t, dec_state, enc_hiddens, enc_hidden_projections, enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t
    
        combined_outputs = torch.stack(combined_outputs, dim=0)
        probs = F.log_softmax(self.vocab_projection(combined_outputs), dim=-1)
        # Memory cleanup
        del Y
        return probs
        
    def step(self, decoder_input, decoder_state, encoder_hiddens, 
             encoder_hiddens_projection, encoder_masks):
        """
        decoder_input: input for the current time step (bs, embedding_dim + hidden_size)
        decoder_state: tuple of hidden states from the previous time step, 
        decoder_state[0] -> previous hidden state (bs, h_s), 
        decoder_state[1] -> previous cell state (bs, h_s)
        encoder_hiddens: encoder's hidden states for all timesteps. (bs, max_seq_len, 2 * h_s)
        encoder_hiddens_projection: encoder hidden states projections (from 2*h_s to h_s)
        encoder_masks: mask for the source sentences (bs, src_len)
        """
        
        new_decoder_state = self.lstm_cell(decoder_input, decoder_state)
        new_decoder_hidden, new_decoder_cell = new_decoder_state
        e_t = torch.bmm(encoder_hiddens_projection, 
                        new_decoder_hidden.unsqueeze(dim=2)).squeeze(dim=2)
        # Fill the attention scores matrix with -inf with given mask positions.
        if encoder_masks is not None:
            e_t.data.masked_fill_(encoder_masks, -float('inf'))
        # Attention probs.
        a_t_probs = nn.functional.softmax(e_t, dim=1)
        a_t = torch.bmm(a_t_probs.unsqueeze(1), encoder_hiddens).squeeze(1) # (bs, 2*h_s)
        u_t = torch.cat([a_t, new_decoder_hidden], dim=1) # (2, 3*h_s)
        v_t = self.combined_output_projection(u_t) # (2, h_s)
        o_t = self.dropout(torch.tanh(v_t)) # (2, h_s)
        # Memory cleanup
        del a_t, u_t, v_t
        return new_decoder_state, o_t, a_t_probs
        
        
class Seq2Seq(nn.Module):
    """
    Sequence to sequence architecture implementation.
    """
    
    def __init__(self, vocabs, embedding_dim, hidden_size, num_layers=1, 
                 bidirectional=True, dropout_p=0.1, device=torch.device("cpu")):
        super(Seq2Seq, self).__init__()
        
        self.vocabs = vocabs
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_p = dropout_p
        self.encoder = Encoder(vocabs.src, embedding_dim, hidden_size, 
                               num_layers=num_layers, bidirectional=bidirectional)
        self.decoder = Decoder(vocabs.tgt, embedding_dim, hidden_size, device=device)
        self.device = device

    def forward(self, src_sents, tgt_sents):
        
        src_lengths = torch.tensor([len(sent) for sent in src_sents])
        src_tensor = to_tensor(self.vocabs.src, src_sents, device=self.device) # (max_seq_len, bs)
        tgt_tensor = to_tensor(self.vocabs.tgt, tgt_sents, device=self.device) # (max_seq_len, bs)
        
        encoder_hiddens, decoder_initial_states = self.encoder(src_tensor, src_lengths)
        encoder_hidden_masks = generate_sent_masks(encoder_hiddens, src_lengths, device=self.device)
        probs = self.decoder(encoder_hiddens, encoder_hidden_masks, 
                                        decoder_initial_states, tgt_tensor)
        
        # Masks for the padded indices in the target sentence.
        tgt_masks = (tgt_tensor != self.vocabs.tgt.pad_idx).float()
        # We skip the <sos> token for the target sentences
        probs = torch.gather(probs, index=tgt_tensor[1:].unsqueeze(-1), dim=-1).squeeze(-1) * tgt_masks[1:]
        scores = probs.sum(dim=0)
        return scores

    def beam_search(self, src_sent, beam_size=5, max_decoding_time_step=70):
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = to_tensor(self.vocabs.src, [src_sent], device=self.device)

        src_encodings, dec_init_vec = self.encoder(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.decoder.attn_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.decoder.hidden_size, device=self.device)

        eos_id = self.vocabs.tgt.eos_idx

        hypotheses = [['<sos>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocabs.tgt.w2i[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.decoder.embedding(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            # def step(self, decoder_input, decoder_state, encoder_hiddens, 
            #          encoder_hiddens_projection, encoder_masks):
            (h_t, cell_t), att_t, _ = self.decoder.step(x, h_tm1,
                                                exp_src_encodings, exp_src_encodings_att_linear, encoder_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.decoder.vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocabs.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocabs.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocabs.tgt.i2w[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '<eos>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    def save(self, path):
        
        params = {
            'args': dict(embedding_dim=self.embedding_dim, 
                         hidden_size=self.decoder.hidden_size, 
                         num_layers=self.num_layers,
                         bidirectional=self.bidirectional,
                         dropout_p=self.dropout_p,
                         device=self.device),
            'vocabs': self.vocabs,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)
    
    @staticmethod
    def load(path):
      
      params = torch.load(path, map_location=lambda storage, loc: storage)
      args = params["args"]
      model = Seq2Seq(vocabs=params["vocabs"], **args)
      model.load_state_dict(params["state_dict"])
      return model