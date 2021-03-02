from .trie import *
import numpy as np
import math
import collections
import torch
import arpa

import _pickle as pickle
from multiprocessing import Pool

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'TrieNode':
            from .trie import TrieNode
            return TrieNode
        return super().find_class(module, name)

class CTCBeamSearchCustom:

    def __init__(self, labels, model_path=None, alpha=0.5, beta=0.5, cutoff_top_n=40, cutoff_prob=-2.1, beam_width=64,
                 blank_id=0, space_id=60, vocab=None, trie_path=None):
        self.NEG_INF = -float("inf")
        self.labels = labels
        self.model_path = model_path
        self.beam_size = beam_width
        self.alpha = alpha
        self.beta = beta
        self.blank_id = blank_id
        self.cutoff_top_n = cutoff_top_n
        self.cutoff_prob = cutoff_prob
        self.vocab = vocab
        self.space_id = space_id
        self.lm = arpa.loadf(self.model_path)[0]
        self.trie_path = trie_path
        self.trie_root = CustomUnpickler(open(self.trie_path, 'rb')).load()
        #with open(self.trie_path, 'rb') as input:
        #    self.trie_root = pickle.load(input)
        #self.p = Process()
        #self.pool = Pool()

    def make_new_beam(self):
        fn = lambda: (self.NEG_INF, self.NEG_INF)
        return collections.defaultdict(fn)

    def logsumexp(self, *args):
        """
        Stable log sum exp.
        """
        if all(a == self.NEG_INF for a in args):
            return self.NEG_INF
        a_max = max(args)
        lsp = math.log(sum(math.exp(a - a_max)
                           for a in args))
        return a_max + lsp

    def apply_lm_word(self, prefix):
        #word = ""
        #for index in reversed(prefix[:-1]):
        #    if index == self.space_id:
        #        break
        #    word += self.labels[index]
        #print("Word : ", word)
        #if len(word)>0 and word.replace(' ', '') in self.vocab:
        #    print(word)   
        #return 0.9 if len(word.replace(' ', ''))>0 and word.replace(' ', '') in self.vocab else 1.08
        return 0.75 if len(prefix.rsplit(" ", 1)) == 2 and prefix.rsplit(" ", 1)[1] in self.vocab else 1.25
    def apply_lm(self, prefix):
        sentence = ""
        for index in prefix[:-1]:
            sentence += self.labels[index]
        #return self.apply_lm_word(sentence)*self.alpha*self.lm.log_p(sentence)
        #print("Sentence: ",len(sentence))
        return self.alpha*self.lm.log_p(sentence)
    def apply_wfst_char(self, n_prefix):
        word = []
        for index in reversed(n_prefix):
            if index == self.space_id:
                break 
            word = [self.labels[index]] + word
        #print(word,probability(self.trie_root, word)) 
        #if np.log(probability(self.trie_root, word)) > 0:
        #    print(np.log(probability(self.trie_root, word)))
        #    print(word)       
        return self.alpha*np.log(probability(self.trie_root, word))

        
    def _decode(self, probs):
        """
        Performs inference for the given output probabilities.
        Arguments:
          probs: The output probabilities (e.g. post-softmax) for each
            time step. Should be an array of shape (time x output dim).
          beam_size (int): Size of the beam to use during inference.
          blank (int): Index of the CTC blank label.
        Returns the output label sequence and the corresponding negative
        log-likelihood estimated by the decoder.
        """
        pred = []
        T, S = probs.shape
        # probs = np.log(probs)

        # Elements in the beam are (prefix, (p_blank, p_no_blank))
        # Initialize the beam with the empty sequence, a probability of
        # 1 for ending in blank and zero for ending in non-blank
        # (in log space).

        beam = [(tuple(), (0.0, self.NEG_INF))]

        for t in range(T): # Loop over time
            # A default dictionary to store the next step candidates.
            next_beam = self.make_new_beam()

            cut_of_value_index = (probs[utterance][t] > self.cutoff_prob).nonzero().reshape(-1) #self.cutoff_prob
            #cut_of_length = min(len(cut_of_value_index), self.cutoff_top_n)
            cut_of_length = self.cutoff_top_n
            top_n_index = torch.argsort(probs[utterance][t], descending=True)[:cut_of_length]

            for s in top_n_index: # Loop over vocab
            #for s in range(S):
                s = int(s)
                p = probs[utterance, t, s]

                # The variables p_b and p_nb are respectively the
                # probabilities for the prefix given that it ends in a
                # blank and does not end in a blank at this time step.
                for prefix, (p_b, p_nb) in beam: # Loop over beam

                    # If we propose a blank the prefix doesn't change.
                    # Only the probability of ending in blank gets updated.
                    if s == self.blank_id:
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_b = self.logsumexp(n_p_b, p_b + p, p_nb + p)
                        next_beam[prefix] = (n_p_b, n_p_nb)
                        continue

                    # Extend the prefix by the new character s and add it to
                    # the beam. Only the probability of not ending in blank
                    # gets updated.
                    end_t = prefix[-1] if prefix else None
                    n_prefix = prefix + (s,)
                    n_p_b, n_p_nb = next_beam[n_prefix]
                    if s != end_t:
                        n_p_nb = self.logsumexp(n_p_nb, p_b + p, p_nb + p)
                    else:
                        # We don't include the previous probability of not ending
                        # in blank (p_nb) if s is repeated at the end. The CTC
                        # algorithm merges characters not separated by a blank.
                        n_p_nb = self.logsumexp(n_p_nb, p_b + p)

                    # *NB* this would be a good place to include an LM score.
                    #if s == self.space_id and s != end_t and end_t is not None:
                        #print("Before npn: ", n_p_nb)
                        #n_p_nb = n_p_nb + self.apply_lm(n_prefix)
                        #print("After npn: ", n_p_nb)
                    
                    if s != self.space_id and end_t != self.space_id and end_t is not None:
                        n_p_nb = n_p_nb + self.apply_wfst_char(n_prefix)
                    #else:
                    #    n_p_nb = n_p_nb -1.5
                    next_beam[n_prefix] = (n_p_b, n_p_nb)

                    # If s is repeated at the end we also update the unchanged
                    # prefix. This is the merging case.
                    if s == end_t:
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_nb = self.logsumexp(n_p_nb, p_nb + p)
                        next_beam[prefix] = (n_p_b, n_p_nb)
                    #if n_p_nb > 0:
                    #    print(n_p_nb)

            # Sort and trim the beam before moving on to the
            # next time-step.
            #for word level self.beta*(x[0].count(self.space_id)+1), char: self.beta*len(x[0]) 
            beam = sorted(next_beam.items(),
                          key=lambda x : self.logsumexp(*x[1])+self.beta*(len(x[0])), 
                          reverse=True)
            beam = beam[:self.beam_size]

        best = beam[0]
        # print("Best : ", best[0])
        # pred.append(best[0])
        return best[0], -self.logsumexp(*best[1])
    def decode(self, probs):
        with Pool() as pool:
            return pool.map(self._decode, probs)
        
