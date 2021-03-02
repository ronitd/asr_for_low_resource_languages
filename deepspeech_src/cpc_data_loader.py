import os 
from torch.utils.data.sampler import Sampler 

from math import log, ceil, floor
import librosa
import numpy as np
import torch
import math
import argparse 
import csv

from torch.utils.data import DataLoader 
from torch.utils.data import Dataset

from python_speech_features import logfbank, mfcc, delta
import scipy.io.wavfile as wav

def nextPowerOf2(n): 
  
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n 


def stack_frame(fbank_feat):
    no_of_frames = fbank_feat.shape[0]
    downsampling = 3
    no_of_stacked_frames = 7
    stack_frames = []
    for frame in range(0, no_of_frames, downsampling):
        if frame + no_of_stacked_frames < no_of_frames:
            stack_frames.append(fbank_feat[frame:frame + no_of_stacked_frames])
    stack_frames = torch.stack(stack_frames)
    return stack_frames


def extract_logmel_unsupervised(path):
    y, s = librosa.load(path, sr=16000)
    fbank_feat = logfbank(y, s, nfilt=80)
    return fbank_feat

def extract_logmelfbank(path): 
    #(rate, sig) = wav.read(path)
    sig, rate = librosa.load(path, sr=16000)
    fbank_feat = logfbank(sig, rate, nfilt=80)
    return fbank_feat

def read_and_trim_audio(path, truncated): 
    raw_audio, _ = librosa.load(path, sr=16000)
    if truncated:
        multiples_of_160 = len(raw_audio)//160
        raw_audio = raw_audio[:multiples_of_160*160]
    return raw_audio

def extract_mfccs(path): 
    raw_audio, sr = librosa.load(path)
    #print("Raw audio length: {}".format(len(raw_audio)))
    mfccs = librosa.feature.mfcc(raw_audio, sr=sr, n_mfcc=13, hop_length=int(0.01 * sr), n_fft=int(0.025 * sr))
    mfcc_delta = librosa.feature.delta(mfccs, order =1 )
    mfcc_delta_delta = librosa.feature.delta(mfccs, order = 2)
    #print("MFCC: {} -- MFCC_Delta: {} -- MFCC_DDelta: {}".format(mfccs.shape, mfcc_delta.shape, mfcc_delta_delta.shape))
    mfccs_overall = np.concatenate((mfccs, mfcc_delta, mfcc_delta_delta), axis=0)
    return mfccs_overall.transpose((1,0))

def get_mfcc(raw_audio, sr):
    mfccs = librosa.feature.mfcc(raw_audio, sr=sr, n_mfcc=13, hop_length=int(0.01 * sr), n_fft=int(0.025 * sr))
    mfcc_delta = librosa.feature.delta(mfccs, order=1)
    mfcc_delta_delta = librosa.feature.delta(mfccs, order=2)
    # print("MFCC: {} -- MFCC_Delta: {} -- MFCC_DDelta: {}".format(mfccs.shape, mfcc_delta.shape, mfcc_delta_delta.shape))
    mfccs_overall = np.concatenate((mfccs, mfcc_delta, mfcc_delta_delta), axis=0)
    return mfccs_overall.transpose((1, 0))


def speed_and_pitch_augumentation_extract_mfcc(speed_factors, pitch_changes, path):
    speed_pitch_mfcc = []
    raw_audio, sr = librosa.load(path)
    bins_per_octave = 24
    # speed_pitch_mfcc.append(get_mfcc(raw_audio, sr))
    for speed_factor in speed_factors:
        y = librosa.effects.time_stretch(raw_audio, speed_factor)
        speed_pitch_mfcc.append(get_mfcc(y, sr))

    for pitch_change in pitch_changes:
        y = librosa.effects.pitch_shift(raw_audio, sr, n_steps=pitch_change, bins_per_octave=bins_per_octave)
        speed_pitch_mfcc.append((get_mfcc(y, sr)))
    return speed_pitch_mfcc


def extract_mfccs_fast(path):
    (rate, sig) = wav.read(path)
    mfcc_feat = mfcc(sig, rate, numcep=13)
    mfcc_delta = delta(mfcc_feat, 9)
    mfcc_delta_delta = delta(mfcc_delta, 9)
    mfccs_overall = np.concatenate((mfcc_feat, mfcc_delta, mfcc_delta_delta), axis=1)
    return mfccs_overall

def read_and_pad_audio(path, min_len): 
    raw_audio, _ = librosa.load(path, sr=16000)
    if (len(raw_audio)) < min_len:
        pad_length = min_len - len(raw_audio)
        raw_audio = np.pad(raw_audio, (0, pad_length), 'constant')
    return raw_audio

def custom_collate(batch):
    """
    Each batch is a list of tuples: (audio, transcript)
    Returns: a tuple containing:
        - inputs: tensor [batch x max_seq_length (of batch) x 1]
        - targets: tensor (Int): all the targets, concatenated into a 1D tensor
        - input_percentages: tensor (Float) [batch]: what percentage of each sample in minibatch is real audio
        - target_sizes: tensor (Int) [batch]: what are the transcript length for each sample in the batch
    """
    def func(p):
        return len(p[0])
    #Each batch consists of multiple samples.
    #Sample[0]: audio ---- Sample[1]: transcript
    batch = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)
    longest_sample = max(batch, key=func)[0]
    minibatch_size = len(batch)
    max_seqlength = len(longest_sample)
    #The input shall have shape: [batch x audio_length x 1]
    inputs = torch.zeros(minibatch_size, max_seqlength, 1)
    #Contains lengths of target outputs
    target_sizes = torch.IntTensor(minibatch_size)
    #Contains all the target output, concatenated into a list
    #Probably because the variable nature of output sequence length -- this is actual smart
    targets = []
    #Contains number indicating what percentage of the max seq is the actual seq
    input_percentages = torch.FloatTensor(minibatch_size)
    duration = []
    print("Here")
    for i in range(minibatch_size):
        sample = batch[i]
        raw_audio = sample[0]
        transcript = sample[1]
        seq_length = len(raw_audio)
        inputs[i].narrow(0, 0, seq_length).copy_(torch.reshape(torch.from_numpy(raw_audio), (-1, 1)))
        target_sizes[i] = len(transcript)
        targets.extend(transcript)
        input_percentages[i] = seq_length / float(max_seqlength)
        
        duration[i] = librosa.get_duration(y=raw_audio, sr=16000)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes, duration

def logmel_collate(batch):
    def func(p):
        return len(p[0])
    #print("batch", batch[:])
    input_mixup = batch[0][2]
    #batch = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)
    longest_sample = max(batch, key=func)[0]
    minibatch_size = len(batch)
    if input_mixup:
    	max_seqlength = len(longest_sample)*3
    else:
        max_seqlength = len(longest_sample)
    #The input shall have shape: [batch x audio_length x mfcc_features*3]
    #max_seqlength = nextPowerOf2(max_seqlength) + 14
    #inputs = torch.zeros(minibatch_size, max_seqlength, 3 * 13)
    inputs = torch.zeros(minibatch_size, max_seqlength, 80)
    #Contains lengths of target outputs
    target_sizes = torch.IntTensor(minibatch_size)
    #Contains all the target output, concatenated into a list
    #Probably because the variable nature of output sequence length -- this is actual smart
    targets = []
    targets_prev_impl = [] 	
    language_attribute = []
    stacked_input = []
    len_transcript = []
    #Contains number indicating what percentage of the max seq is the actual seq
    input_percentages = torch.FloatTensor(minibatch_size)

    random_samples = np.random.randint(minibatch_size, size=minibatch_size)
    random_samples_1 = np.random.randint(minibatch_size, size=minibatch_size)
    for i in range(minibatch_size):
        sample = batch[i]
        raw_audio = sample[0]  #seq x features
        if input_mixup:
            raw_audio_1 = np.append(sample[0], batch[random_samples[i]][0], axis=0)
            raw_audio = np.append(raw_audio_1, batch[random_samples_1[i]][0], axis=0)
        transcript = sample[1]
        #print(len(sample[1]))  
        if input_mixup:       
            transcript.extend(batch[random_samples[i]][1])
            transcript.extend(batch[random_samples_1[i]][1])             
        seq_length = len(raw_audio)

        #inputs[i].narrow(0, 0, seq_length).copy_(torch.reshape(torch.from_numpy(raw_audio), (-1, 3 * 13)))
					
        inputs[i].narrow(0, 0, seq_length).copy_(torch.reshape(torch.from_numpy(raw_audio), (-1, 80)))
        stacked_input.append(stack_frame(inputs[i]))
        target_sizes[i] = len(transcript)
        targets.append(torch.LongTensor(transcript))
        targets_prev_impl.extend(transcript)	
        input_percentages[i] = seq_length / float(max_seqlength)
        language_attribute.append(sample[3])
    #print(targets)
    #targets = torch.stack(targets)
	
    targets = torch.nn.utils.rnn.pad_sequence(targets, padding_value = 0)
    trg_mask = targets == 0
    #print("trg mask shape", trg_mask.shape)
    #print("Mask trg ", trg_mask.T)	
    targets_prev_impl = torch.IntTensor(targets_prev_impl) 
     	
    #print("Targets Shape: ", targets.shape)
    #print("Target: ", targets)  
    #print("Input Shape: ", inputs.shape)
    language_attribute = torch.stack(language_attribute)
    stacked_inputs = torch.stack(stacked_input)
    stacked_inputs = stacked_inputs.view(stacked_inputs.shape[0], stacked_inputs.shape[1], -1)
    #exit() 
  	
    input_mask = (stacked_inputs == torch.zeros(80*7))
    #print("Mask Shape: ", input_mask.shape)	
    #print("Stacked Input: ", stacked_inputs.shape)
    input_mask = torch.sum(input_mask, dim=2) 
    #print("After Sum Mask Shape: ", input_mask.shape)
    input_mask = input_mask != 0
    #print("After Sum Mask Shape: ", input_mask.shape)
    #print("Input Mask: ", input_mask)	
    #exit() 	   
    return inputs, targets, input_percentages, target_sizes, language_attribute, targets_prev_impl, input_mask, trg_mask 

def mfcc_collate(batch): 
    def func(p):
        return len(p[0])
    #print("batch", batch[:])
    input_mixup = batch[0][2]
    #batch = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)
    #for (x, y) in batch:
    #    print(y)
    #print(batch[0][0])
    #print(batch[0][1])
    #print(batch[0][2])
    #print(batch[0][3]) 
    #exit() 
    longest_sample = max(batch, key=func)[0]
    minibatch_size = len(batch)
    if input_mixup:
    	max_seqlength = len(longest_sample)*3
    else:
        max_seqlength = len(longest_sample)
    #The input shall have shape: [batch x audio_length x mfcc_features*3]
    max_seqlength = nextPowerOf2(max_seqlength) + 14
    inputs = torch.zeros(minibatch_size, max_seqlength, 3 * 13)
    # inputs = torch.zeros(minibatch_size, max_seqlength, 80)
    #Contains lengths of target outputs
    target_sizes = torch.IntTensor(minibatch_size)
    #Contains all the target output, concatenated into a list
    #Probably because the variable nature of output sequence length -- this is actual smart
    targets = []
    language_attribute = []
    #Contains number indicating what percentage of the max seq is the actual seq
    input_percentages = torch.FloatTensor(minibatch_size)

    random_samples = np.random.randint(minibatch_size, size=minibatch_size)
    random_samples_1 = np.random.randint(minibatch_size, size=minibatch_size)
    for i in range(minibatch_size):
        sample = batch[i]
        raw_audio = sample[0]  #seq x features
        if input_mixup:
            raw_audio_1 = np.append(sample[0], batch[random_samples[i]][0], axis=0)
            raw_audio = np.append(raw_audio_1, batch[random_samples_1[i]][0], axis=0)
        #print(sample[0].shape)
        #print(batch[random_samples[i]][0].shape)
        #print(batch[random_samples_1[i]][0].shape)
        #print(raw_audio.shape)
        #exit()  
        transcript = sample[1]
        #print(len(sample[1]))  
        if input_mixup:       
            transcript.extend(batch[random_samples[i]][1])
            transcript.extend(batch[random_samples_1[i]][1])
             
        
        #print(len(batch[random_samples[i]][1]))
        #print(len(batch[random_samples_1[i]][1]))
        #print(len(transcript))
        #exit()
        seq_length = len(raw_audio)

        inputs[i].narrow(0, 0, seq_length).copy_(torch.reshape(torch.from_numpy(raw_audio), (-1, 3 * 13)))
					
        # inputs[i].narrow(0, 0, seq_length).copy_(torch.reshape(torch.from_numpy(raw_audio), (-1, 80)))
        target_sizes[i] = len(transcript)
        targets.extend(transcript)
        #print(transcript) 
        input_percentages[i] = seq_length / float(max_seqlength)
        #print(sample[3])
        language_attribute.append(sample[3])
    targets = torch.IntTensor(targets)
    #print(language_attribute)
    #print(len(language_attribute))
    try:
        language_attribute = torch.stack(language_attribute)
    except:
        print(language_attribute)   
    #print("Collate: ", language_attribute.shape)
    return inputs, targets, input_percentages, target_sizes

def self_supervise_collate(batch): 
    def func(p):
        return len(p)

    # print("Batch shape: {}".format(len(batch)))
    # print("Batch [0]: {}".format(batch[0].shape))
    #print("Batch: {}".format(batch))
    batch = sorted(batch, key=lambda sample: len(sample), reverse=True)

    shortest_sample = min(batch, key=func)
    minibatch_size = len(batch)
    min_seqlength = len(shortest_sample)
    if min_seqlength > 100000:
        min_seqlength = 100000

    inputs = torch.zeros(minibatch_size, min_seqlength, 80)

    for i in range(minibatch_size):
        sample = batch[i]
        seq_length = len(sample)

        #Here's where we should do the cropping
        if len(sample) > min_seqlength:
            cropped_start = np.random.randint(low=0, high=seq_length-min_seqlength, size=1)[0]
        else:
            cropped_start = 0
        cropped_sample = sample[cropped_start:cropped_start+min_seqlength]
        inputs[i].narrow(0,0,min_seqlength).copy_(torch.reshape(torch.from_numpy(cropped_sample), (-1, 80)))
    return inputs

class RawAudioDataset(Dataset): 
    def __init__(self, wav_filenames, transcripts, labels, truncated=False):
        self.wav_filenames = wav_filenames
        self.transcripts = transcripts
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self.truncated = truncated

    def __len__(self):
        return len(self.wav_filenames)

    def __getitem__(self, index):
        wav_filename = self.wav_filenames[index]
        transcript = self.transcripts[index]
        transcript = self.parse_transcript(transcript)
        wav_file_raw = read_and_trim_audio(wav_filename, truncated=self.truncated)
        print("Here")
        print(librosa.get_duration(wav_file_raw))
        exit()
        return (wav_file_raw, transcript)

    def parse_transcript(self, transcript):
        """Convert transcript into a list of integers"""
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

class RawAudioDataLoader(DataLoader): 
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for RawAudioDatasets
        """
        super(RawAudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = custom_collate

class RawAudioBucketingSampler(Sampler): 
    def __init__(self, data_source, batch_size=1):
        super(RawAudioBucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source))) #Each item in the dataset gets an id
        #Generating bins -- each bin is from i to i+batch.
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self):
        np.random.shuffle(self.bins)

class RawAudioDataset_minlength(Dataset): 
    def __init__(self, wav_filenames, transcripts, labels):
        self.wav_filenames = wav_filenames
        self.transcripts = transcripts
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])

    def __len__(self):
        return len(self.wav_filenames)

    def parse_transcript(self, transcript):
        """Convert transcript into a list of integers"""
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

    def __getitem__(self, index):
        wav_filename = self.wav_filenames[index]
        transcript = self.transcripts[index]
        transcript = self.parse_transcript(transcript)
        wav_file_raw = read_and_pad_audio(wav_filename, 9600)
        return (wav_file_raw, transcript)


class MFCCDataset(Dataset):
    def __init__(self, wav_filenames, transcripts, labels, languages_attribute = [], input_mixup=False):
        self.wav_filenames = wav_filenames
        self.transcripts = transcripts
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self.input_mixup = input_mixup
        self.languages_attribute = languages_attribute
     

    def __len__(self):
        return len(self.wav_filenames)

    def __getitem__(self, index):
        """
        mfccs have shape <features x seq_len>
        """
        wav_filename = self.wav_filenames[index]
        transcript = self.transcripts[index]
        transcript = self.parse_transcript(transcript)
        mfccs = extract_mfccs(wav_filename)
        # mfccs = extract_mfccs_fast(wav_filename)
        if len(self.languages_attribute) > 0:
            languages_attribute = self.languages_attribute[index]
        else:
            language_attribute = -1
		
        return (mfccs, transcript, self.input_mixup, language_attribute)

    def parse_transcript(self, transcript):
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript


class MFCCDatasetRonit(Dataset):
    def __init__(self, wav_filenames, transcripts, labels):
        self.wav_filenames = wav_filenames
        self.transcripts = transcripts
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self.mfcc = []
        for wav_filename in self.wav_filenames:
            self.mfcc.append(extract_mfccs(wav_filename))


    def __len__(self):
        return len(self.wav_filenames)

    def __getitem__(self, index):
        """
        mfccs have shape <features x seq_len>
        """
        # wav_filename = self.wav_filenames[index]
        transcript = self.transcripts[index]
        transcript = self.parse_transcript(transcript)
        # mfccs = extract_mfccs(wav_filename)
        # mfccs = extract_mfccs_fast(wav_filename)
        mfccs = self.mfcc[index]
        return (mfccs, transcript)

    def parse_transcript(self, transcript):
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript


class MFCCDatasetRonitSpeedAug(Dataset):

    def __init__(self, wav_filenames, transcripts, labels):
        speed_factors = [0.93, 1.07]
        pitch_changes = [-2, 2]
        aug_length = len(speed_factors) + len(pitch_changes)
        self.wav_filenames = wav_filenames*(aug_length)
        self.transcripts = transcripts*(aug_length)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self.mfcc = []
        augumented_data = [[] for i in range(aug_length)]
        for wav_filename in wav_filenames:
            mfccs = speed_and_pitch_augumentation_extract_mfcc(speed_factors, pitch_changes, wav_filename)
            index = 0
            for i in mfccs:
                # if index == -1:
                #     self.mfcc.append(i)
                # else:
                augumented_data[index].append(i)
                index += 1
        for i in augumented_data:
            self.mfcc = self.mfcc + i
    def __len__(self):
        return len(self.wav_filenames)

    def __getitem__(self, index):
        """
        mfccs have shape <features x seq_len>
        """
        transcript = self.transcripts[index]
        transcript = self.parse_transcript(transcript)
        mfccs = self.mfcc[index]
        return (mfccs, transcript)

    def parse_transcript(self, transcript):
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

class PrecomputedMFCCDataset(Dataset):
    def __init__(self, mfcc_paths, transcripts, labels):
        self.mfcc_paths = mfcc_paths
        self.transcripts = transcripts
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])

    def __len__(self):
        return len(self.mfcc_paths)

    def __getitem__(self, index):
        mfcc_path = self.mfcc_paths[index]
        transcript = self.transcripts[index]
        transcript = self.parse_transcript(transcript)
        mfccs = np.load(mfcc_path)
        return (mfccs, transcript)

    def parse_transcript(self, transcript):
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

class MFCCDataLoader(DataLoader): 
    def __init__(self, *args, **kwargs):
        super(MFCCDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = mfcc_collate

class LogMelDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(LogMelDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = logmel_collate

class MFCCBucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        super(MFCCBucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self):
        np.random.shuffle(self.bins)

class LogMelDataset(Dataset):
    def __init__(self, wav_filenames, transcripts, labels, languages_attribute, input_mixup=False):
        self.wav_filenames = wav_filenames
        self.transcripts = transcripts
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self.input_mixup = input_mixup
        self.languages_attribute = languages_attribute

    def __len__(self):
        return len(self.wav_filenames)

    def __getitem__(self, index):
        wav_filename = self.wav_filenames[index]
        transcript = "<" + self.transcripts[index] + ">"
        transcript = self.parse_transcript(transcript)
        filterbank = extract_logmelfbank(wav_filename)
        return (filterbank, transcript, self.input_mixup, self.languages_attribute[index])

    def parse_transcript(self, transcript):
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

class LogMelUnlabeledDataset(Dataset):
    def __init__(self, wav_filenames):
        self.wav_filenames = wav_filenames

    def __len__(self):
        return len(self.wav_filenames)

    def __getitem__(self, index):
        wav_filename = self.wav_filenames[index]
        filterbank = extract_logmel_unsupervised(wav_filename)
        return filterbank

class LogMelSelfSuperviseDataLoader(DataLoader): 
    def __init__(self, *args, **kwargs):
        super(LogMelSelfSuperviseDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self_supervise_collate

class LogMelSelfSuperviseBucketingSampler(Sampler): 
    def __init__(self, data_source, batch_size=1):
        super(LogMelSelfSuperviseBucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self):
        np.random.shuffle(self.bins)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--alphabet", type=str)

    args = parser.parse_args()

    labels = set()
    with open(args.alphabet, 'r') as f:
        for row in f:
            labels.add(row[0])
    labels = sorted(labels)
    labels_str = ""
    for label in labels:
        labels_str += label

    print("Label string: {}".format(labels_str))

    raw_audio_paths = []
    transcripts = []
    with open(args.test_csv) as f:
        temp_data = []
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            wav_path = row[0]
            if (wav_path != "wav_filename"):
                temp_data.append(row)
                #transcript = row[2]
                #raw_audio_paths.append(wav_path)
                #transcripts.append(transcript)
        #Sort by wavsize, which is row[1]
        temp_data = sorted(temp_data, key=lambda k: int(k[1]))
        for temp_datum in temp_data:
            raw_audio_paths.append(temp_datum[0])
            transcripts.append(temp_datum[2])
            #print("Size: {}".format(temp_datum[1]))

    #dataset = RawAudioDataset(raw_audio_paths, transcripts, labels_str)
    dataset = MFCCDataset(raw_audio_paths, transcripts, labels_str)
    print("Data set length: {}".format(len(dataset)))
    print("========= INDIVIDUAL DATA LENGTH ====================")
    for i in range(len(dataset)):
        #print("i = {} -- len: {}".format(i, len(dataset[i][0])))
        print("i = {} -- shape: {}".format(i, dataset[i][0].shape))
        #print("Transcript: {}".format(transcripts[i]))

    # sampler = RawAudioBucketingSampler(dataset, batch_size=args.batch_size)
    # data_loader = RawAudioDataLoader(dataset, num_workers=0, batch_sampler=sampler)
    sampler = MFCCBucketingSampler(dataset, batch_size=args.batch_size)
    data_loader = MFCCDataLoader(dataset, num_workers=0, batch_sampler=sampler)

    for i, (data) in enumerate(data_loader):
        if i == len(sampler):
            break
        inputs, targets, input_percentages, target_sizes = data
        print("===========================================")
        print("Iter: {}".format(i))
        print("Inputs shape: {}".format(inputs.shape))
        print("Targets: {}".format(targets))
        print("Targets shape: {}".format(targets.shape))
        print("Input perc: {}".format(input_percentages))
        print("Target sizes: {}".format(target_sizes))
        print("Sum of target sizes: {}".format(torch.sum(target_sizes)))


