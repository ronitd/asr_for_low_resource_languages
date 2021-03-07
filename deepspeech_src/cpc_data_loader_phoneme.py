import os
from torch.utils.data.sampler import Sampler
# import torchaudio
import librosa
import numpy as np
import torch
import math
import argparse
import csv
import re
import os
# from specAugment import spec_augment_pytorch
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


def extract_logmel_unsupervised(path):
    y, s = librosa.load(path, sr=16000)
    fbank_feat = logfbank(y, s, nfilt=80)
    return fbank_feat


def preprocessed(path):
    logmelfbank = np.load(path)
    mfcc = np.load(path.replace("logmelfbank", "mfcc"))
    # print("MFCC: ", mfcc.shape)
    S_DB = np.load(path.replace("logmelfbank", "spectogram"))
    # print("S_DB: ", S_DB.shape)
    fbank_feat_padded = np.pad(logmelfbank, [(0, mfcc.shape[0] - logmelfbank.shape[0]), (0, 0)], mode='constant')
    # print("fbank : ", fbank_feat_padded.shape)

    # S_DB_MFCC_SPECTOGRAM = np.concatenate((S_DB, mfcc, fbank_feat_padded), axis=1)
    S_DB_MFCC_SPECTOGRAM = np.concatenate((mfcc, fbank_feat_padded), axis=1)
    return S_DB_MFCC_SPECTOGRAM


def savenumpy(value, path, name):
    pattern = "/"
    new_path = path[:path.rfind(pattern, 0, path.rfind(pattern)) + 1] + name + path[path.rfind(pattern):path.rfind(
        ".")] + ".npy"
    # if not os.path.isfile(new_path):
    np.save(new_path, value)


def speed_spectogram(path):
    y, sr = librosa.load(path, sr=16000)
    bins_per_octave = 24
    speed_factor = np.random.uniform(low=0.9, high=1.1)
    y = librosa.effects.time_stretch(y, speed_factor)
    return logfbank(y, sr, nfilt=80)


def extract_logmelfbank(path):
    y, sr = librosa.load(path, sr=16000)
    # sig, rate = librosa.load(path)
    fbank_feat = logfbank(y, sr, nfilt=80)
    savenumpy(fbank_feat, path, 'logmelfbank')
    mfcc = extract_mfccs(path)
    savenumpy(mfcc, path, 'mfcc')
    sr = 16000
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=int(0.025 * sr), hop_length=int(0.01 * sr), n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max).transpose((1, 0))

    savenumpy(S_DB, path, 'spectogram')
    # log_fbank_mfcc = np.concatenate((mfcc, S_DB), axis=1)

    '''
    print("Librosa: ", fbank_feat)
    
    waveform, sample_rate = torchaudio.load_wav(path)
    fbank_feat = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80)
    print("Kaldi: ", fbank_feat)
    exit()
    '''

    return S_DB


def extract_logmelfbank_mfcc(wav_filename):
    # print("Here"*5)
    mfcc = extract_mfccs(wav_filename)
    logmelfbank = extract_logmelfbank(wav_filename)
    # print(mfcc.shape)
    # print(logmelfbank.shape)
    a = np.pad(logmelfbank, [(0, mfcc.shape[0] - logmelfbank.shape[0]), (0, 0)], mode='constant')
    # print(a.shape)
    log_fbank_mfcc = np.concatenate((a, mfcc), axis=1)

    ''' 
    #Spectogram
    path = wav_filename
    y, sr = librosa.load(path, sr=16000)
    sr = 16000
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=int(0.025 * sr), hop_length=int(0.01 * sr), n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    log_fbank_mfcc = np.concatenate((mfcc, S_DB), axis=1)

    path = wav_filename
    pattern = "/"
    new_path = path[:path.rfind(pattern, 0, path.rfind(pattern)) + 1] + "logmelfbank"+ path[path.rfind(pattern):path.rfind(".")] + ".npy"
    np.save(new_path, logmelfbank)
    
    return logmelfbank
    #print("-"*100)
    '''
    return log_fbank_mfcc


def read_and_trim_audio(path, truncated):
    raw_audio, _ = librosa.load(path, sr=16000)
    if truncated:
        multiples_of_160 = len(raw_audio) // 160
        raw_audio = raw_audio[:multiples_of_160 * 160]
    return raw_audio


def extract_mfccs(path):
    raw_audio, sr = librosa.load(path, sr=16000)
    # print("SR: ", sr)
    # print("Raw audio length: {}".format(len(raw_audio)))
    mfccs = librosa.feature.mfcc(raw_audio, sr=sr, n_mfcc=13, hop_length=int(0.01 * sr), n_fft=int(0.025 * sr))
    mfcc_delta = librosa.feature.delta(mfccs, order=1)
    mfcc_delta_delta = librosa.feature.delta(mfccs, order=2)
    # print("MFCC: {} -- MFCC_Delta: {} -- MFCC_DDelta: {}".format(mfccs.shape, mfcc_delta.shape, mfcc_delta_delta.shape))
    mfccs_overall = np.concatenate((mfccs, mfcc_delta, mfcc_delta_delta), axis=0)
    return mfccs_overall.transpose((1, 0))


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


def logmel_mfcc_collate(batch):
    def func(p):
        return len(p[0])

    batch = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)
    longest_sample = max(batch, key=func)[0]
    minibatch_size = len(batch)
    max_seqlength = len(longest_sample)
    # max_seqlength =  nextPowerOf2(len(longest_sample))
    # The input shall have shape: [batch x audio_length x mfcc_features*3]
    # inputs = torch.zeros(minibatch_size, max_seqlength, 3 * 13)
    inputs = torch.zeros(minibatch_size, max_seqlength, 128 + 39 + 80)
    # Contains lengths of target outputs
    target_sizes = torch.IntTensor(minibatch_size)
    # Contains all the target output, concatenated into a list
    # Probably because the variable nature of output sequence length -- this is actual smart
    targets = []
    # Contains number indicating what percentage of the max seq is the actual seq
    input_percentages = torch.FloatTensor(minibatch_size)

    for i in range(minibatch_size):
        sample = batch[i]
        raw_audio = sample[0]  # seq x features
        # print(raw_audio)
        transcript = sample[1]
        seq_length = len(raw_audio)
        # inputs[i].narrow(0, 0, seq_length).copy_(torch.reshape(torch.from_numpy(raw_audio), (-1, 3 * 13)))
        inputs[i].narrow(0, 0, seq_length).copy_(torch.reshape(torch.from_numpy(raw_audio), (-1, 128 + 39 + 80)))
        # inputs[i].narrow(0, 0, seq_length).copy_(torch.reshape(raw_audio, (-1, 80)))
        target_sizes[i] = len(transcript)
        targets.extend(transcript)
        input_percentages[i] = seq_length / float(max_seqlength)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes


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

    # Each batch consists of multiple samples.
    # Sample[0]: audio ---- Sample[1]: transcript
    batch = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)
    longest_sample = max(batch, key=func)[0]
    minibatch_size = len(batch)
    max_seqlength = len(longest_sample)
    # The input shall have shape: [batch x audio_length x 1]
    inputs = torch.zeros(minibatch_size, max_seqlength, 1)
    # Contains lengths of target outputs
    target_sizes = torch.IntTensor(minibatch_size)
    # Contains all the target output, concatenated into a list
    # Probably because the variable nature of output sequence length -- this is actual smart
    targets = []
    # Contains number indicating what percentage of the max seq is the actual seq
    input_percentages = torch.FloatTensor(minibatch_size)

    for i in range(minibatch_size):
        sample = batch[i]
        raw_audio = sample[0]
        transcript = sample[1]
        seq_length = len(raw_audio)
        inputs[i].narrow(0, 0, seq_length).copy_(torch.reshape(torch.from_numpy(raw_audio), (-1, 1)))
        target_sizes[i] = len(transcript)
        targets.extend(transcript)
        input_percentages[i] = seq_length / float(max_seqlength)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes


def logmel_collate(batch):
    def func(p):
        return len(p[0])

    batch = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)
    longest_sample = max(batch, key=func)[0]
    minibatch_size = len(batch)
    max_seqlength = len(longest_sample)
    # max_seqlength =  nextPowerOf2(len(longest_sample))
    # The input shall have shape: [batch x audio_length x mfcc_features*3]
    # inputs = torch.zeros(minibatch_size, max_seqlength, 3 * 13)
    inputs = torch.zeros(minibatch_size, max_seqlength, 128)
    # Contains lengths of target outputs
    target_sizes = torch.IntTensor(minibatch_size)
    # Contains all the target output, concatenated into a list
    # Probably because the variable nature of output sequence length -- this is actual smart
    targets = []
    # Contains number indicating what percentage of the max seq is the actual seq
    input_percentages = torch.FloatTensor(minibatch_size)

    for i in range(minibatch_size):
        sample = batch[i]
        raw_audio = sample[0]  # seq x features
        # print(raw_audio)
        transcript = sample[1]
        seq_length = len(raw_audio)
        # inputs[i].narrow(0, 0, seq_length).copy_(torch.reshape(torch.from_numpy(raw_audio), (-1, 3 * 13)))
        inputs[i].narrow(0, 0, seq_length).copy_(torch.reshape(torch.from_numpy(raw_audio), (-1, 128)))
        # inputs[i].narrow(0, 0, seq_length).copy_(torch.reshape(raw_audio, (-1, 39)))
        target_sizes[i] = len(transcript)
        targets.extend(transcript)
        input_percentages[i] = seq_length / float(max_seqlength)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes


def mfcc_collate(batch):
    def func(p):
        return len(p[0])

    batch = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)
    longest_sample = max(batch, key=func)[0]
    minibatch_size = len(batch)
    max_seqlength = len(longest_sample)
    # The input shall have shape: [batch x audio_length x mfcc_features*3]
    inputs = torch.zeros(minibatch_size, max_seqlength, 3 * 13)
    # inputs = torch.zeros(minibatch_size, max_seqlength, 80)
    # Contains lengths of target outputs
    target_sizes = torch.IntTensor(minibatch_size)
    # Contains all the target output, concatenated into a list
    # Probably because the variable nature of output sequence length -- this is actual smart
    targets = []
    # Contains number indicating what percentage of the max seq is the actual seq
    input_percentages = torch.FloatTensor(minibatch_size)

    for i in range(minibatch_size):
        sample = batch[i]
        raw_audio = sample[0]  # seq x features
        # print(raw_audio)
        transcript = sample[1]
        seq_length = len(raw_audio)
        inputs[i].narrow(0, 0, seq_length).copy_(torch.reshape(torch.from_numpy(raw_audio), (-1, 3 * 13)))
        # inputs[i].narrow(0, 0, seq_length).copy_(torch.reshape(torch.from_numpy(raw_audio), (-1, 80)))
        target_sizes[i] = len(transcript)
        targets.extend(transcript)
        input_percentages[i] = seq_length / float(max_seqlength)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes


def self_supervise_collate(batch):
    def func(p):
        return len(p)

    # print("Batch shape: {}".format(len(batch)))
    # print("Batch [0]: {}".format(batch[0].shape))
    # print("Batch: {}".format(batch))
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

        # Here's where we should do the cropping
        if len(sample) > min_seqlength:
            cropped_start = np.random.randint(low=0, high=seq_length - min_seqlength, size=1)[0]
        else:
            cropped_start = 0
        cropped_sample = sample[cropped_start:cropped_start + min_seqlength]
        inputs[i].narrow(0, 0, min_seqlength).copy_(torch.reshape(torch.from_numpy(cropped_sample), (-1, 80)))
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
        ids = list(range(0, len(data_source)))  # Each item in the dataset gets an id
        # Generating bins -- each bin is from i to i+batch.
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
    def __init__(self, wav_filenames, transcripts, labels):
        self.wav_filenames = wav_filenames
        self.transcripts = transcripts
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])

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
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self):
        np.random.shuffle(self.bins)


class LogMelDataset(Dataset):
    def __init__(self, wav_filenames, transcripts, labels, lexicon, use_preprocessed=False, is_train=False):

        self.wav_filenames = wav_filenames
        # self.transcripts = transcripts
        self.len = len(wav_filenames)
        self.labels_map = labels
        self.lexicon = lexicon
        self.phoneme_to_index = {v: k for k, v in labels.items()}
        self.use_preprocessed = use_preprocessed
        self.features = []
        self.labels = []
        self.is_train = is_train
        for i in range(len(wav_filenames)):
            if not is_train:
                wav_filename = wav_filenames[i].replace("Audios16KHz", "spectogram")

                wav_filename = wav_filename.replace("wav", "npy")
                feature = np.load(wav_filename) if self.use_preprocessed else extract_logmelfbank(wav_filename)
                self.features.append(feature)

            transcript = transcripts[i]
            transcript = self.parse_transcript(transcript)
            self.labels.append(transcript)

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        if self.is_train:
            wav_filename = self.wav_filenames[index]
            wav_filename = wav_filename.replace("Audios", "spectogram")
            wav_filename = wav_filename.replace("wav", "npy")
            # filterbank = np.load(wav_filename) if self.use_preprocessed else extract_logmelfbank(wav_filename)
            try:
                filterbank = np.load(wav_filename) if self.use_preprocessed else extract_logmelfbank(wav_filename)
            except:
                print(wav_filename)
                print("Here")
        # filterbank = extract_logmelfbank(self.wav_filenames[index])
        else:
            filterbank = self.features[index]

        # transcript = self.transcripts[index]
        # transcript = self.parse_transcript(transcript, wav_filename)
        # filterbank = speed_spectogram(wav_filename)
        # filterbank = test(wav_filename)

        # return (filterbank, transcript)
        return (filterbank, self.labels[index])

    def parse_transcript(self, transcript, filename="a"):
        org_transcript = transcript

        transcript = re.split("( )", transcript.rstrip())
        t = []
        for x in transcript:
            if x in self.lexicon:
                t.extend(self.lexicon.get(x))
            else:
                t.extend([x])
        t = list(filter(None, [self.phoneme_to_index[t[i]] for i in range(len(t))]))
        '''
        try:
            t = list(filter(None, [self.phoneme_to_index[t[i]] for i in range(len(t))]))
        except:
            print("Transcript: ", org_transcript)
            print(filename)
            t = [0]
        '''
        return t


class LogMelTriphoneDataset(Dataset):
    def __init__(self, wav_filenames, transcripts, word_to_triphone_int, use_preprocessed=False):
        self.wav_filenames = wav_filenames
        self.transcripts = transcripts
        self.word_to_triphone_int = word_to_triphone_int
        self.use_preprocessed = use_preprocessed
        #print(word_to_triphone_int)

    def __len__(self):
        return len(self.wav_filenames)

    def __getitem__(self, index):
        wav_filename = self.wav_filenames[index]
        wav_filename = wav_filename.replace("Audios16KHz", "spectogram")
        wav_filename = wav_filename.replace("wav", "npy")
        transcript = self.transcripts[index]
        transcript = self.parse_transcript(transcript)
        filterbank = np.load(wav_filename) if self.use_preprocessed else extract_logmelfbank_mfcc(wav_filename)
        return (filterbank, transcript)

    def parse_transcript(self, transcript):
        transcript = re.split("( )", transcript.rstrip())
        t = []
        for x in transcript:
            if x in self.word_to_triphone_int:
                t.extend(self.word_to_triphone_int.get(x))
            else:
                t.extend([None])
                print("Word Not in dict: ", x)
        # t = list(filter(None, [self.phoneme_to_index[t[i]] for i in range(len(t))]))
        return t


class LogMelMFCCDataset(Dataset):
    def __init__(self, wav_filenames, transcripts, labels, lexicon, use_preprocessed=False):
        self.wav_filenames = wav_filenames
        self.transcripts = transcripts
        self.labels_map = labels
        self.lexicon = lexicon
        self.phoneme_to_index = {v: k for k, v in labels.items()}
        self.use_preprocessed = use_preprocessed

    def __len__(self):
        return len(self.wav_filenames)

    def __getitem__(self, index):
        wav_filename = self.wav_filenames[index]
        transcript = self.transcripts[index]
        transcript = self.parse_transcript(transcript)
        filterbank = np.load(wav_filename) if self.use_preprocessed else extract_logmelfbank_mfcc(wav_filename)
        return (filterbank, transcript)

    def parse_transcript(self, transcript):
        transcript = re.split("( )", transcript)
        t = []
        for x in transcript:
            if x in self.lexicon:
                t.extend(self.lexicon.get(x))
            else:
                t.extend([x])
        t = list(filter(None, [self.phoneme_to_index[t[i]] for i in range(len(t))]))
        return t


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
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

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
        # transcript = row[2]
        # raw_audio_paths.append(wav_path)
        # transcripts.append(transcript)
        # Sort by wavsize, which is row[1]
        temp_data = sorted(temp_data, key=lambda k: int(k[1]))
        for temp_datum in temp_data:
            raw_audio_paths.append(temp_datum[0])
            transcripts.append(temp_datum[2])
    # print("Size: {}".format(temp_datum[1]))

    # dataset = RawAudioDataset(raw_audio_paths, transcripts, labels_str)
    dataset = MFCCDataset(raw_audio_paths, transcripts, labels_str)
    print("Data set length: {}".format(len(dataset)))
    print("========= INDIVIDUAL DATA LENGTH ====================")
    for i in range(len(dataset)):
        # print("i = {} -- len: {}".format(i, len(dataset[i][0])))
        print("i = {} -- shape: {}".format(i, dataset[i][0].shape))
    # print("Transcript: {}".format(transcripts[i]))

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
