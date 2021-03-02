import torch 
from .utils import levenshtein_distance
import numpy as np
import collections
import math

class Decoder(object): 
	def __init__(self, labels, blank_index=0):
		''' 
		self.labels = labels 
		self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
		self.blank_index = blank_index 
		space_index = len(labels)
		if not ('_' in labels):
			labels = labels + '_' 
		if ' ' in labels: 
			space_index = labels.index(' ') 
		if '_' in labels: 
			self.blank_index = labels.index('_')
		self.space_index = space_index 
		print("Space index: {}".format(self.space_index))
		print("Blank index: {}".format(self.blank_index))
		'''
		
		self.labels = labels
		self.blank_index = blank_index
		space_index = len(labels)
		print("In Decoder label: ", labels)
		self.int_to_phoneme = {v: k for k, v in self.labels.items()}
		if not ('_' in self.int_to_phoneme):
			labels[len(labels)] = '_'
		 
		if ' ' in self.int_to_phoneme: 
			space_index = self.int_to_phoneme[' '] 
		if '_' in self.int_to_phoneme: 
			self.blank_index = self.int_to_phoneme['_']
		
		self.space_index = space_index 
		print("Space index: {}".format(self.space_index))
		print("Blank index: {}".format(self.blank_index))
		

	def wer(self, s1, s2): 
		# b = set(s1.split() + s2.split())
		# word2char = dict(zip(b, range(len(b))))

		# w1 = [chr(word2char[w]) for w in s1.split()]
		# w2 = [chr(word2char[w]) for w in s2.split()]

		# return Lev.distance(''.join(w1), ''.join(w2))
		#print("S1", s1)
		#print("S1", s1.split(self.space_index))
		#print("S2", s2)
		#print("S2", s2.split(self.space_index))
		#print("S1 shape: ", s1.shape)
		#print("S2 shape: ", s2.shape)
		#print("S2 shape: ", s2.split(self.space_index).shape)
		#exit()
		lev_distance = levenshtein_distance(s1.split(" "), s2.split(" "))
		return lev_distance/len(s2.split()), lev_distance, len(s2.split())

	def cer(self, s1, s2): 
		#s1, s2 = s1.replace(' ', ''), s2.replace(' ', '') 
		#return Lev.distance(s1, s2)
		#print("CER")
		#print("S1: ", s1)
		#print("S2: ", s2)
		lev_distance = levenshtein_distance(s1, s2)
		return lev_distance/len(s2), lev_distance, len(s2)

	def int_to_str(self, int_arr): 
		out_str = "" 
		for i in int_arr: 
			out_str += self.int_to_char[i]
		return out_str

	def decoder(self, probs, sizes=None):
		raise NotImplementedError

class GreedyDecoder(Decoder): 
	def __init__(self, labels, blank_index=0): 
		super(GreedyDecoder, self).__init__(labels, blank_index)

	def convert_to_strings(self, sequences, sizes=None, remove_repetitions=False, return_offsets=False):
		strings = [] 
		offsets = [] if return_offsets else None 
		for x in range(len(sequences)): 
			
			seq_len = sizes[x] if sizes is not None else len(sequences[x]) 
			string, string_offset = self.process_strings(sequences[x], seq_len, remove_repetitions)
			strings.append([string])
			if return_offsets: 
				offsets.append([string_offset])
		if return_offsets: 
			return strings, offsets 
		else: 
			return strings 

	def process_strings(self, sequence, size, remove_repetitions=False): 
		string = '' 
		offsets = [] 
		#print("Sequence shape: {}".format(sequence.shape))
		#print("Size: {}".format(size))
		for i in range(size): 
			#char = self.int_to_char[sequence[i].item()]
			char  = str(int(sequence[i].item()))
			if char != str(self.blank_index): 
				if remove_repetitions and i != 0 and char == str(int(sequence[i - 1].item())):
					pass 
				elif char == str(self.space_index):
					string += ' '
					offsets.append(i) 
				else: 
					string = string + char 
					offsets.append(i)
		return string, torch.tensor(offsets, dtype=torch.int) 

	def decode(self, probs, sizes=None):
		"""
		Shape of probs should be: <batch x seq x class>
		"""
		#print("Probs shape: {}".format(probs.shape)) 
		_, max_probs = torch.max(probs, 2) 
		#print("Max prob shape: {}".format(max_probs.shape))
		#print("Sizes: {}".format(sizes))
		#return max_probs, True
		strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes, 
			remove_repetitions=True, return_offsets=True)
		return strings, offsets 

class BeamCTCDecoder(Decoder): 
	def __init__(self, labels, ctc_labels=None ,lm_path=None, alpha=0, beta=0, cutoff_top_n=25, 
		cutoff_prob=-2.1, beam_width=100, num_processes=4, blank_index=0, log_probs_input=True, phoneme_vocab=None, trie=None): 
		super(BeamCTCDecoder, self).__init__(labels)
		self.ctc_labels = ctc_labels
		from .CTCBeamSearchCustom import CTCBeamSearchCustom
		try: 
			from ctcdecode import CTCBeamDecoder 
		except ImportError: 
			raise ImportError("BeamCTCDecoder requires ctcdecoder package")
		# self._decoder = CTCBeamDecoder(labels.lower(), lm_path, alpha, beta, cutoff_top_n, 
		# 	cutoff_prob, beam_width=beam_width, num_processes=num_processes)
		#print("Labels Decoder: ", labels)
		self._decoder = CTCBeamDecoder(ctc_labels, lm_path, alpha, beta, cutoff_top_n, 
			cutoff_prob, beam_width=beam_width, num_processes=num_processes, log_probs_input= False)
		#self._decoder = CTCBeamSearchCustom(labels, lm_path, alpha, beta, cutoff_top_n, 
		#	cutoff_prob, beam_width=beam_width, vocab = phoneme_vocab, trie_path=trie)
	
	def convert_to_strings(self, out, seq_len): 
		results = [] 
		for b, batch in enumerate(out): 
			utterances = [] 
			for p, utt in enumerate(batch): 
				size = len(utt)
				#size = seq_len[p]
				#print("utt: {}".format(utt))
				#print("utt size: {}".format(utt.size)) 
				if size > 0: 
					transcript = "".join(map(lambda x: self.int_to_char[x.item()] if x.item() in self.int_to_char.keys() else "", utt[0:size]))
				else: 
					transcript = "" 
				utterances.append(utterances)
			results.append(utterances)	
		return results 

	def convert_tensor(self, offsets, sizes): 
		results = [] 
		for b, batch in enumerate(offsets): 
			utterances = [] 
			for p, utt in enumerate(batch): 
				size = sizes[b][p]
				if sizes[b][p] > 0: 
					utterances.append(utt[0:size]) 
				else: 
					utterances.append(torch.tensor([], dtype=torch.int))
			results.append(utterances) 
		return results 

	def custom_convert_to_string(self, tokens, vocab, seq_lens):
		#return ''.join([vocab[x] for x in tokens[0:seq_len]])
		
		strings = [] 
		#print("Tokens shape: {}".format(tokens.shape))
		for i in range(len(tokens)):
			#Batches
			token = tokens[i][0]
			seq_len = seq_lens[i][0]
			#print("i = {} - Token shape: {} -- Seq len: {}".format(i, token.shape, seq_len))
			#for x in token[0:seq_len]:
			#	print(x)	
			decoded_string = ''.join([vocab[x.item()] for x in token[0:seq_len]])
			strings.append(decoded_string)
		return strings
	
	def custom_convert_to_string_ronit(self, tokens, vocab, seq_lens):
		#return ''.join([vocab[x] for x in tokens[0:seq_len]])
		
		strings = [] 
		for i in range(len(tokens)):
			#Batches
			token = tokens[i]
			#seq_len = seq_lens[i]
			#print("i = {} - Token shape: {} -- Seq len: {}".format(i, token.shape, seq_len))
			#for x in token[0:seq_len]:
			#	print(x)	
			decoded_string = ''.join([vocab[x] for x in token])
			strings.append(decoded_string)
		return strings

	def decode(self, probs, sizes=None): 
		probs = torch.pow(np.exp(1), probs).cpu()
		out, scores, offsets, seq_lens = self._decoder.decode(probs, sizes)
		#Custom
		#out, scores  = self._decoder.decode(probs)
		#offsets = 0
		
		#print("After decode")
		#print("Seq lens shape: {}".format(seq_lens.shape))
		#strings = self.convert_to_strings(out, seq_lens)
		#strings = self.custom_convert_to_string_ronit(out, self.labels, seq_lens = None)
		strings = self.custom_convert_to_string(out, self.labels, seq_lens)
		offsets = self.convert_tensor(offsets, seq_lens)
		return strings, offsets

	def convert_to_strings_target(self, sequences, sizes=None, remove_repetitions=False, return_offsets=False):
		strings = [] 
		offsets = [] if return_offsets else None 
		for x in range(len(sequences)): 
			seq_len = sizes[x] if sizes is not None else len(sequences[x]) 
			string, string_offset = self.process_strings(sequences[x], seq_len, remove_repetitions)
			strings.append([string])
			if return_offsets: 
				offsets.append([string_offset])
		if return_offsets: 
			return strings, offsets 
		else: 
			return strings 

	def process_strings(self, sequence, size, remove_repetitions=False): 
		string = '' 
		offsets = [] 
		# print("Sequence shape: {}".format(sequence.shape))
		# print("Size: {}".format(size))
		#print("int to phoneme: ", self.int_to_phoneme)
		for i in range(size): 
			char = self.labels[sequence[i].item()]
			if char != self.labels[self.blank_index]: 
				if remove_repetitions and i != 0 and char == self.labels[sequence[i-1].item()]:
					pass 
				elif char == self.labels[self.space_index]:
					string += ' '
					offsets.append(i) 
				else: 
					string = string + char 
					offsets.append(i)
		return string, torch.tensor(offsets, dtype=torch.int)