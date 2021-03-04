import argparse
import os
import random
import time
import csv
import torch
import sys

import numpy as np
from tqdm import tqdm
# from torch_baidu_ctc import ctc_loss, CTCLoss
from deepspeech_src.cpc_data_loader_phoneme import MFCCDataset, MFCCBucketingSampler, MFCCDataLoader, \
    PrecomputedMFCCDataset, RawAudioDataset, RawAudioBucketingSampler, RawAudioDataLoader, LogMelDataset, \
    LogMelDataLoader, LogMelTriphoneDataset
from deepspeech_src.decoder_ronit import GreedyDecoder, BeamCTCDecoder
# from deepspeech_src.model import ResNextASR
from deepspeech_src.resnext_v2_model import ResNextASR_v2, Encoder, Decoder
# from deepspeech_src.test_model import ResNextASR_v2


from deepspeech_src.utils_ronit import check_loss, read_csv_deepspeech, read_label_file, read_csv_mfccs, load_obj
from deepspeech_src.logger import TensorboardLogger

# from specAugment import spec_augment_pytorch


parser = argparse.ArgumentParser(description="DeepSpeech training With CPC")
# Input data CSVs
parser.add_argument("--train_csv", help="Path to CSV containing training utterances")
parser.add_argument("--val_csv", help="Path to CSV containing validation utterances")
parser.add_argument("--test_csv", help="Path to CSV containing test utterances")
parser.add_argument("--use_preprocessed", default=False, action="store_true", help="Use pre-processed logmelfbank")

# Model hyper params
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_workers", default=1, type=int, help="Number of worker threads used in data loading")
parser.add_argument("--labels", type=str, help="Path to a dictionary file containing all labels")
parser.add_argument("--lexicon", type=str, help="Path to a lexicon")

# parser.add_argument("--alphabet", type=str, help="Path to a txt file containing all tokens in the alphabet")
# parser.add_argument("--hidden_size", default=512, type=int, help="Hidden size of RNNs")
# parser.add_argument("--hidden_layers", default=5, type=int, help="Number of RNN layers")
# parser.add_argument("--rnn-type", default='gru', help="Type of RNN. Either rnn|gru|lstm") 
parser.add_argument("--epochs", default=70, type=int, help="Number of training epochs")
parser.add_argument("--lr", "--learning-rate", default=3e-4, type=float, help="Initial learning rate")
parser.add_argument("--depth", default=9, type=int, help="Depth of the Model")
parser.add_argument("--depth_residual", default=False, action="store_true", help="Depth Residual")
parser.add_argument("--width", default=8, type=int, help="Width of the Model")
parser.add_argument("--width_jump", default=4, type=int, help="Difference between kernel size of the Model")
parser.add_argument("--dense_dim", type=int, default=256)
parser.add_argument("--bottleneck_depth", type=int, default=16)

parser.add_argument("--weight_standardization", default=False, action="store_true", help="To do Weight Standardization")
parser.add_argument("--group_norm", default=False, action="store_true", help="To do Group Normalization")

# CUDA
parser.add_argument("--no_cuda", default=False, action="store_true", help="Use CPU instead of GPU")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU Id to use")

# Checkpoints and finetuning
parser.add_argument("--checkpoint_dir", default="./checkpoint/", type=str, help="Directory to store checkpoint")
parser.add_argument("--final_model_path", type=str, help="Path to store final training model",
                    default="./wav2letter_Inception.pth")
parser.add_argument("--continue_from", type=str, help="Path to checkpoint to start from")
parser.add_argument("--finetune", default=False, action="store_true", help="Set to finetune model from continue_from")

# Shuffling dataset and sortaGrad
parser.add_argument("--no_shuffle", default=False, action="store_true",
                    help="Turn off shuffling and sample from dataset based on sequence length (smallest to largest)")
parser.add_argument("--no_sorta_grad", default=False, action="store_true",
                    help="'Turn off ordering of dataset on sequence length for the first epoch")

# CPC model 
parser.add_argument("--cpc_model_path", type=str, help="Path to the CPC model")
# parser.add_argument("--finetune_cpc", default=False, action="store_true", help="Set to finetune CPC")

parser.add_argument("--lm_path", type=str, help="Path to the language model")
parser.add_argument("--train", default=False, action="store_true")
parser.add_argument("--test", default=False, action="store_true")
parser.add_argument("--use_beamsearch", default=False, action="store_true")
parser.add_argument("--transfer", default=False, action="store_true")
parser.add_argument("--old_alphabet", type=str)
parser.add_argument("--alphabet", type=str, help="Require for Beam Search")
parser.add_argument("--ctc_labels", type=str, help="Require for Beam Search")
parser.add_argument("--val_preprocessed", default=False, action="store_true")
parser.add_argument("--lm_alpha", type=float, default=0.75)
parser.add_argument("--lm_beta", type=float, default=1.85)
parser.add_argument("--beam_width", type=int, default=100)
parser.add_argument("--sweep_decode", default=False, action="store_true")
parser.add_argument("--log_dir", type=str, default="./logs/")
parser.add_argument("--phoneme_vocab", type=str, help="Require for Beam Search")
parser.add_argument("--trie", type=str, help="Require for Beam Search")
parser.add_argument("--triphone_to_int", type=str, help="triphone to int mapping")
parser.add_argument("--int_to_triphone", type=str, help="int to triphone mapping")
parser.add_argument("--word_to_triphone_label", type=str, help="Word to int(labels) mapping")

seed = 4
# np.random.seed(seed)
random.seed(seed)
# os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)


def to_np(x):
    """Convert tensor x to numpy array"""
    return x.cpu().numpy()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    args = parser.parse_args()

    if args.train:
        # torch.backends.cudnn.deterministic = True
        train_paths, train_transcripts = read_csv_deepspeech(args.train_csv, True)
        val_paths, val_transcripts = read_csv_deepspeech(args.val_csv, False)

        device = torch.device("cuda" if not args.no_cuda else "cpu")
        use_gpu_id = args.gpu_id  # Set which GPU ID to use

        if torch.cuda.is_available():
            if torch.cuda.device_count == 1:
                use_gpu_id = args.gpu_id

        torch.cuda.set_device(use_gpu_id)

        save_folder = args.checkpoint_dir
        os.makedirs(save_folder, exist_ok=True)

        loss_results, cer_results, wer_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(
            args.epochs)

        best_wer = None
        avg_loss, start_epoch, start_iter, optim_state = 0, 0, 0, None

        # TODO: Implement a way to load model
        # Read labels:
        labels = load_obj(args.labels)
        print("labels: {}".format(labels))

        if args.transfer:
            old_labels = read_label_file(args.old_alphabet)
            if ('_') not in old_labels:
                print("Adding CTC blank to old labels")
                old_labels = '_' + old_labels
            print("Old labels: {}".format(old_labels))

        if args.transfer:

            model = ResNextASR_v2(
                num_features=128,
                num_classes=len(old_labels),
                dense_dim=256,
                bottleneck_depth=16,
                args=args
            )
        else:
            model = ResNextASR_v2(
                num_features=128,
                num_classes=len(labels),
                dense_dim=args.dense_dim,
                bottleneck_depth=args.bottleneck_depth,
                args=args
            )

        decoder = GreedyDecoder(labels)

        if args.cpc_model_path is None:
            split = 3000

            end = 6999
            # end =20500

            '''
            train_dataset_short = LogMelDataset(train_paths[:split], train_transcripts[:split], labels, args.use_preprocessed)
            train_dataset_medium = LogMelDataset(train_paths[split: 2*split], train_transcripts[split:2*split], labels, args.use_preprocessed)
            train_dataset_long = LogMelDataset(train_paths[2*split:end], train_transcripts[2*split:end], labels, args.use_preprocessed)
            '''

            train_dataset = LogMelTriphoneDataset(train_paths, train_transcripts, load_obj(args.word_to_triphone_label),
                                          args.use_preprocessed)
            val_dataset = LogMelTriphoneDataset(val_paths, val_transcripts, load_obj(args.word_to_triphone_label),
                                        args.use_preprocessed)

            '''
            train_sampler_short  = MFCCBucketingSampler(train_dataset_short, batch_size=args.batch_size)
            train_sampler_medium = MFCCBucketingSampler(train_dataset_medium, batch_size=args.batch_size)
            train_sampler_long  = MFCCBucketingSampler(train_dataset_long, batch_size=args.batch_size)
            '''

            train_sampler_all = MFCCBucketingSampler(train_dataset, batch_size=args.batch_size)

            '''
            train_loader_short = LogMelDataLoader(train_dataset_short, num_workers=args.num_workers, batch_sampler=train_sampler_short)
            train_loader_medium = LogMelDataLoader(train_dataset_medium, num_workers=args.num_workers, batch_sampler=train_sampler_medium)
            train_loader_long = LogMelDataLoader(train_dataset_long, num_workers=args.num_workers, batch_sampler=train_sampler_long)
            '''

            train_loader_all = LogMelDataLoader(train_dataset, num_workers=args.num_workers,
                                                batch_sampler=train_sampler_all)

            '''
            train_sampler = MFCCBucketingSampler(train_dataset, batch_size=args.batch_size)
            train_loader = LogMelDataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler)
            '''

            val_sampler = MFCCBucketingSampler(val_dataset, batch_size=args.batch_size)
            val_loader = LogMelDataLoader(val_dataset, num_workers=args.num_workers, batch_sampler=val_sampler)
        else:

            train_dataset = RawAudioDataset(train_paths, train_transcripts, labels)
            val_dataset = RawAudioDataset(val_paths, val_transcripts, labels)

            train_sampler = RawAudioBucketingSampler(train_dataset, batch_size=args.batch_size)
            train_loader = RawAudioDataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler)

            val_sampler = RawAudioBucketingSampler(val_dataset, batch_size=args.batch_size)
            val_loader = RawAudioDataLoader(val_dataset, num_workers=args.num_workers, batch_sampler=val_sampler)

        if (args.continue_from is not None):
            print("Loading model from: {}".format(args.continue_from))
            package = torch.load(args.continue_from)
            state_dict = package['state_dict']
            model.load_state_dict(state_dict)

        if args.transfer:
            print("Changing the last layer to do transfer learning")
            model.classifier = torch.nn.Conv1d(256, len(labels), kernel_size=1)
            model.classifier.apply(model.init_weights)
        # print(device)
        # exit()
        model = model.to(device)
        parameters = model.parameters()
        optimizer = torch.optim.Adam(parameters, lr=args.lr)

        # optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, nesterov=True)

        if optim_state is not None:
            optimizer.load_state_dict(optim_state)

        print("Acoustic Model: ")
        print(model)
        print("Number of parameters: %d" % ResNextASR_v2.get_param_size(model))
        # Documentation: https://pypi.org/project/torch-baidu-ctc/
        # To use this, we feed: criterion(x, y, xs, ys) where:
        # x has shape: <T x N x D> (max timestep x batch x num_classes)
        # y is a 1D tensor (int32) of concatenated target
        # xs contains the length of each x in the sample (1D tensor, each element = len(x[:, n, :]))
        # ys contains the length of each target (sum(ys) = len(y))
        criterion = torch.nn.CTCLoss(blank=decoder.blank_index)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        lowest_loss = 999999999
        lowest_cer = 999999999
        lowest_wer = 999999999

        best_epoch = 0

        # Create a tensorboard logging thingy
        logger = TensorboardLogger(args.log_dir)
        step = 0
        # if args.cpc_model_path is None:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            end = time.time()
            start_epoch_time = time.time()

            # if epoch != 0:
            # 	train_sampler.shuffle()
            '''
            if epoch > 250:
                train_loader = train_loader_short
                train_sampler = train_sampler_short
                 #train_sampler.shuffle()
            elif epoch > 500:
                train_loader = train_loader_medium
                train_sampler = train_sampler_medium
            elif epoch > 750:
                train_loader = train_loader_long
                train_sampler = train_sampler_long
            else:
            '''
            train_loader = train_loader_all
            train_sampler = train_sampler_all
            for i, (data) in enumerate(train_loader, start=start_iter):
                if i == len(train_sampler):
                    # We have reach the end of the epoch
                    break
                # Input: <batch x audio_len x 1>
                inputs, targets, input_percentages, target_sizes = data
                # Determine the individual input size by multiplying the percentage by the input size
                # input_sizes = input_percentages.mul_(int(inputs.size(1))).int()
                # print("Pre CPC input sizes: {}".format(input_sizes))
                # input_sizes_np = to_np(input_sizes)
                # input_sizes_np = (np.floor_divide(input_sizes_np, 160))
                # input_sizes = torch.from_numpy(input_sizes_np).int()

                data_time.update(time.time() - end)
                inputs = inputs.float().to(device)

                # print("After CPC: {}".format(context.shape))
                # out: [batch x seq_len // 160 x alphabet size]
                # Note: currently using input_lengths --- maybe we can use input_sizes?
                # out, output_sizes = model(inputs.transpose(1,2), input_sizes)
                # print("Inputs shape: {}".format(inputs.shape))

                # inputs = spec_augment_pytorch.spec_augment(inputs, time_warping_para=40, frequency_masking_para=27, time_masking_para=70, frequency_mask_num=2, time_mask_num=2)
                out = model(inputs)
                output_sizes = torch.full((len(inputs),), out.shape[2], dtype=torch.long)
                # print("Shape of output_sizes: {}".format(output_sizes))
                # out = out.transpose(0, 1) #Now: [seq_len x batch x alphabet size]
                # print("After network: {}".format(out.shape))

                float_out = out.float().cpu()  # Ensure that out is float 32

                # print("Float out shape: {}".format(float_out.shape))
                # print("Targets: {}".format(targets))
                # print("Output sizes: {}".format(output_sizes))
                # print("Target sizes: {}".format(target_sizes))
                # print("Shape of float out: {}".format(float_out.transpose(1,2).transpose(0,1).shape))
                # Float out needs to be <T x N x C>
                loss = criterion(float_out.transpose(1, 2).transpose(0, 1), targets.cpu(), output_sizes.cpu(),
                                 target_sizes.cpu())
                # loss = loss / inputs.size(0)

                loss_value = loss.item()
                valid_loss, error = check_loss(loss, loss_value)
                if valid_loss:
                    optimizer.zero_grad()
                    # Compute gradient
                    loss.backward()
                    # Cut off gradient at 400 to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 400)
                    optimizer.step()
                else:
                    print(error)
                    print("Skipping grad update")
                    loss_value = 0

                avg_loss += loss_value
                losses.update(loss_value, inputs.size(0))

                batch_time.update(time.time() - end)
                end = time.time()
                if (i % 256 == 0):
                    print("Epoch [{}][{}/{}]\tTime {:.3f} ({:.3f})\tData {:.3f} ({:.3f})\tLoss: {:.4f} ({:.4f})".format(
                        (epoch + 1), (i + 1), len(train_sampler), batch_time.val, batch_time.avg,
                        data_time.val, data_time.avg, losses.val, losses.avg))
                    # Logging
                    logger.scalar_summary('train_loss', losses.val, step + 1)
                step += 1
                del loss, out, float_out
            avg_loss /= len(train_sampler)

            epoch_time = time.time() - start_epoch_time
            # print("===== EPOCH: {}\tTime taken (s): {:.0f}\tAverage loss: {:.4f}\tSpec: {:.4f}\tMFCC: {:.4f}\tfbank: ".format((epoch+1), epoch_time, avg_loss, model.gamma_spectogram.item(), model.gamma_mfcc.item(), model.gamma_fbank.item()))
            print("===== EPOCH: {}\tTime taken (s): {:.0f}\tAverage loss: {:.4f}".format((epoch + 1), epoch_time,
                                                                                         avg_loss))

            # Perform validation
            start_iter = 0
            total_cer, total_wer = 0, 0
            total_words, total_chars, total_words_edit, total_chars_edit = 0, 0, 0, 0
            total_loss = 0
            model.eval()
            with torch.no_grad():
                for i, (data) in enumerate(val_loader):
                    inputs, targets, input_percentages, target_sizes = data
                    # input_sizes = input_percentages.mul_(int(inputs.size(1))).int()
                    inputs = inputs.to(device)

                    # Split the string of target into individual target
                    split_targets = []
                    offset = 0
                    for size in target_sizes:
                        split_targets.append(targets[offset:offset + size])
                        offset += size

                    out = model(inputs)

                    float_out = out.float()

                    output_sizes = torch.full((len(inputs),), out.shape[2], dtype=torch.long)

                    loss = criterion(float_out.transpose(1, 2).transpose(0, 1).cpu(), targets.cpu(), output_sizes.cpu(),
                                     target_sizes.cpu())
                    loss_value = loss.item()
                    # print("Validation CTC Loss for {}/{}: {}".format(i, len(val_loader),loss_value))
                    total_loss += loss_value
                    # print("Out shape: {}".format(out.shape))
                    # print("Output sizes: {}".format(output_sizes))

                    decoded_output, _ = decoder.decode(float_out.transpose(1, 2), output_sizes)
                    target_strings = decoder.convert_to_strings(split_targets)

                    # DEBUG CODE
                    # _, max_probs = torch.max(out.transpose(1,2), 2)
                    # # print("Split targets: {}".format(split_targets))
                    # # print("Target strings: {}".format(target_strings))
                    # #print("Max probs shape: {}".format(max_probs.shape))
                    # for i in range(len(out.transpose(1,2))):
                    # 	print("Max probs at i: {}".format(max_probs[i].data.cpu().numpy()))
                    # 	print("Decoded output {}: {}".format(i, decoder.int_to_str(max_probs[i].data.cpu().numpy())))
                    # 	print("Target string: {}: {}".format(i, decoder.int_to_str(split_targets[i].numpy())))

                    wer, cer = 0, 0
                    for x in range(len(split_targets)):
                        transcript, reference = decoded_output[x][0], target_strings[x][0]
                        word_er, word_edit_dist, total_word = decoder.wer(transcript, reference)
                        char_er, char_edit_dist, total_char = decoder.cer(transcript, reference)
                        total_words += total_word
                        total_chars += total_char
                        total_words_edit += word_edit_dist
                        total_chars_edit += char_edit_dist
                    del out
                wer = total_words_edit / total_words
                cer = total_chars_edit / total_chars
                total_loss = total_loss / len(val_loader)
                wer *= 100
                cer *= 100
                loss_results[epoch] = avg_loss
                wer_results[epoch] = wer
                cer_results[epoch] = cer
                print(
                    "Validation summary epoch: {}\tAverage CTC Loss: {}\tAverage WER: {:.3f}\tAverage CER {:.3f}".format(
                        epoch + 1, total_loss, wer, cer))
                logger.scalar_summary('CER', cer, epoch + 1)
                logger.scalar_summary('WER', wer, epoch + 1)
                logger.scalar_summary('val_loss', total_loss, epoch + 1)

            if (total_loss < lowest_loss or cer < lowest_cer):
                print("Find newest model at epoch: {}. Saving at: {}".format(epoch, args.final_model_path))
                # Save this epoch
                torch.save(ResNextASR_v2.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                                   wer_results=wer_results, cer_results=cer_results),
                           args.final_model_path)

                lowest_loss = total_loss
                lowest_cer = cer

    if args.test:
        print("****************TESTING PRE-TRAINED MODEL ****************************")
        test_paths, test_transcripts = read_csv_deepspeech(args.test_csv, False)
        # test_paths, test_transcripts = read_csv_mfccs(args.test_csv)
        device = torch.device("cuda" if not args.no_cuda else "cpu")

        # TODO: Implement a way to load model
        # Read labels:
        labels = load_obj(args.labels)
        print("labels: {}".format(labels))

        # Read alphabet:
        alphabets = read_label_file(args.alphabet)
        if ('_' not in alphabets):
            print("Adding CTC blank to alphabets")
            alphabets = '_' + alphabets
        print("alphabets: {}".format(alphabets))

        if args.cpc_model_path is not None:
            model = ResNextASR_v2(
                num_features=256,
                num_classes=len(labels),
                dense_dim=256,
                bottleneck_depth=16
            )
        else:
            model = ResNextASR_v2(
                num_features=128,
                num_classes=len(labels),
                dense_dim=args.dense_dim,
                bottleneck_depth=args.bottleneck_depth,
                args=args
            )

        if args.use_beamsearch:
            check = []
            # ctc_labels = load_obj(args.ctc_labels)
            phoneme_vocab = load_obj(args.phoneme_vocab)
            for key in labels:
                check.append(labels[key])
            print("check: ", check)
            check = ['_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                     't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7',
                     ' ', ',']
            print("Len C:", len(check))
            # k= list(check)
            # print("K", k)
            args.lm_path = None
            decoder = BeamCTCDecoder(labels, check, lm_path=args.lm_path, alpha=args.lm_alpha, beta=args.lm_beta,
                                     beam_width=args.beam_width, phoneme_vocab=phoneme_vocab, trie=args.trie)
            print("Intiailized")
        else:
            decoder = GreedyDecoder(labels)

        if args.cpc_model_path is None:
            # test_dataset = MFCCDataset(test_paths, test_transcripts, labels)
            # test_sampler = MFCCBucketingSampler(test_dataset, batch_size=args.batch_size)
            # test_loader = MFCCDataLoader(test_dataset, num_workers=args.num_workers, batch_sampler=test_sampler)
            test_dataset = LogMelDataset(test_paths, test_transcripts, labels, load_obj(args.lexicon),
                                         args.use_preprocessed, False)
            test_sampler = MFCCBucketingSampler(test_dataset, batch_size=args.batch_size)
            test_loader = LogMelDataLoader(test_dataset, num_workers=args.num_workers, batch_sampler=test_sampler)
        else:
            test_dataset = RawAudioDataset(test_paths, test_transcripts, labels)
            test_sampler = RawAudioBucketingSampler(test_dataset, batch_size=args.batch_size)
            test_loader = RawAudioDataLoader(test_dataset, num_workers=args.num_workers, batch_sampler=test_sampler)

        if (args.continue_from is not None):
            print("Loading model from: {}".format(args.continue_from))
            package = torch.load(args.continue_from)
            state_dict = package['state_dict']
            model.load_state_dict(state_dict)
        else:
            print("Please provide a model to test in continue_from")
            sys.exit()

        model = model.to(device)

        parameters = model.parameters()

        # print("DeepSpeech Model: ")
        # print(model)
        print("Number of parameters: %d" % ResNextASR_v2.get_param_size(model))

        # Documentation: https://pypi.org/project/torch-baidu-ctc/
        # To use this, we feed: criterion(x, y, xs, ys) where:
        # x has shape: <T x N x D> (max timestep x batch x num_classes)
        # y is a 1D tensor (int32) of concatenated target
        # xs contains the length of each x in the sample (1D tensor, each element = len(x[:, n, :]))
        # ys contains the length of each target (sum(ys) = len(y))
        criterion = torch.nn.CTCLoss(blank=decoder.blank_index)

        # Initializing CPC model
        if args.cpc_model_path is not None:
            wav2vec = Wav2Vec(
                timestep=12,
                batch_size=args.batch_size,
                z_dim=512,
                c_dim=256
            ).to(device)
            print("Path to CPC: {}".format(args.cpc_model_path))
            wav2vec.load_state_dict(torch.load(args.cpc_model_path)['state_dict'])
            wav2vec.eval()
        if not args.sweep_decode:
            if True:
                # if args.cpc_model_path is None:
                end = time.time()
                start_epoch_time = time.time()

                avg_data_time = 0
                avg_decode_time = 0
                avg_wer_time = 0

                # Perform validation
                total_cer, total_wer = 0, 0
                total_words, total_chars, total_words_edit, total_chars_edit = 0, 0, 0, 0
                total_loss = 0
                model.eval()
                with torch.no_grad():
                    for i, (data) in enumerate(test_loader):
                        inputs, targets, input_percentages, target_sizes = data
                        # input_sizes = input_percentages.mul_(int(inputs.size(1))).int()
                        inputs = inputs.to(device)

                        # Split the string of target into individual target
                        split_targets = []
                        offset = 0
                        for size in target_sizes:
                            split_targets.append(targets[offset:offset + size])
                            offset += size

                        data_time = time.time()

                        # #Put the data through CPC
                        # cpc_hidden = cpc_model.init_hidden(inputs.shape[0], 256, True)
                        # context, hidden = cpc_model.predict(inputs.transpose(1,2), cpc_hidden)

                        # input_sizes_np = to_np(input_sizes)
                        # input_sizes_np = (np.floor_divide(input_sizes_np, 160))
                        # input_sizes = torch.from_numpy(input_sizes_np).int()

                        # Put the CPC output through the DeepSpeech
                        # input_lengths = torch.IntTensor(context.shape[1])
                        if args.cpc_model_path is not None:
                            context = wav2vec.predict(inputs.transpose(1, 2))
                            out = model(context)
                        else:
                            out = model(inputs)
                        # print("Output Shape: ", out.shape)
                        # print("Output : ",out[0])
                        # exit()

                        float_out = out.float()

                        output_sizes = torch.full((len(inputs),), out.shape[2], dtype=torch.long)

                        neural_net_time = time.time()

                        loss = criterion(float_out.transpose(1, 2).transpose(0, 1).cpu(), targets.cpu(),
                                         output_sizes.cpu(), target_sizes.cpu())
                        loss_value = loss.item()
                        # print("Validation CTC Loss for {}/{}: {}".format(i, len(val_loader),loss_value))
                        total_loss += loss_value
                        # print("Out shape: {}".format(out.shape))
                        # print("Output sizes: {}".format(output_sizes))

                        decoded_output, _ = decoder.decode(float_out.transpose(1, 2), output_sizes)
                        if args.use_beamsearch:
                            target_strings = decoder.convert_to_strings_target(split_targets)
                        else:
                            target_strings = decoder.convert_to_strings(split_targets)

                        decode_time = time.time()
                        # DEBUG CODE
                        # _, max_probs = torch.max(out.transpose(1,2), 2)
                        # # print("Split targets: {}".format(split_targets))
                        # # print("Target strings: {}".format(target_strings))
                        # #print("Max probs shape: {}".format(max_probs.shape))
                        # for i in range(len(out.transpose(1,2))):
                        # 	print("Max probs at i: {}".format(max_probs[i].data.cpu().numpy()))
                        # 	print("Decoded output {}: {}".format(i, decoder.int_to_str(max_probs[i].data.cpu().numpy())))
                        # 	print("Target string: {}: {}".format(i, decoder.int_to_str(split_targets[i].numpy())))

                        wer, cer = 0, 0
                        for x in range(len(target_strings)):
                            if (args.use_beamsearch):
                                transcript, reference = decoded_output[x], target_strings[x][0]
                            else:
                                transcript, reference = decoded_output[x][0], target_strings[x][0]
                            if (x == 0):
                                print("Predicted: {}".format(transcript))
                                print("Reference: {}".format(reference))
                            with open('gu-test-pred.txt', 'a+') as pred_file:
                                pred_file.write(transcript + "\n")
                            with open('gu-test-transcript.txt', 'a+') as transcript_file:
                                transcript_file.write(reference + "\n")

                            # wer += decoder.wer(transcript, reference) / float(len(reference.split()))
                            # cer += decoder.cer(transcript, reference) / float(len(reference))
                            # wer += decoder.wer(transcript, reference)
                            # cer += decoder.cer(transcript, reference)
                            word_er, word_edit_dist, total_word = decoder.wer(transcript, reference)
                            char_er, char_edit_dist, total_char = decoder.cer(transcript, reference)
                            total_words += total_word
                            total_chars += total_char
                            total_words_edit += word_edit_dist
                            total_chars_edit += char_edit_dist
                            total_wer += word_er
                            total_cer += char_er

                        wer_time = time.time()
                        # total_wer += wer
                        # total_cer += cer
                        del out
                        avg_data_time += data_time - end
                        avg_decode_time += decode_time - neural_net_time
                        avg_wer_time += wer_time - decode_time
                        if i % 10 == 0:
                            print("{}/{}: Data: {:.3f}({:3f})\tDecode: {:3f}({:3f})\tWER Time: {:3f}({:3f})".format(
                                i, len(test_sampler), data_time - end, avg_data_time / (i + 1),
                                                      decode_time - neural_net_time, avg_decode_time / (i + 1),
                                                      wer_time - neural_net_time, avg_wer_time / (i + 1)))
                        end = time.time()
                    wer = total_words_edit / total_words
                    cer = total_chars_edit / total_chars
                    total_loss = total_loss / len(test_loader)
                    wer *= 100
                    cer *= 100
                    print("TEST RESULT: \tAverage CTC Loss: {:.3f}\tAverage WER: {:.3f}\tAverage CER {:.3f}".format(
                        total_loss, wer, cer))
                    print("Total words edit: {}\tTotal words: {}".format(total_words_edit, total_words))
                    print("Total chars edit: {}\tTotal chars: {}".format(total_chars_edit, total_chars))
                    print("Old metrics: {} \t {}".format(total_wer * 100 / len(test_loader.dataset),
                                                         total_cer * 100 / len(test_loader.dataset)))
        else:
            print("Sweeping decoder params")
            lm_alphas = [0, 0.5, 0.75, 1.0, 1.25, 1.50]
            lm_betas = [0, 0.75, 1.0, 1.25, 1.75, 1.85, 2.10]
            beam_widths = [100, 256, 512, 1024]

            best_params = None
            best_wer = 999999

            for beam_width in beam_widths:
                for lm_alpha in lm_alphas:
                    for lm_beta in lm_betas:
                        decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=lm_alpha, beta=lm_beta,
                                                 beam_width=beam_width)
                        # if args.cpc_model_path is None:
                        end = time.time()
                        start_epoch_time = time.time()

                        avg_data_time = 0
                        avg_decode_time = 0
                        avg_wer_time = 0

                        # Perform validation
                        total_cer, total_wer = 0, 0
                        total_loss = 0
                        model.eval()
                        with torch.no_grad():
                            for i, (data) in enumerate(test_loader):
                                inputs, targets, input_percentages, target_sizes = data
                                # input_sizes = input_percentages.mul_(int(inputs.size(1))).int()
                                inputs = inputs.to(device)

                                # Split the string of target into individual target
                                split_targets = []
                                offset = 0
                                for size in target_sizes:
                                    split_targets.append(targets[offset:offset + size])
                                    offset += size

                                data_time = time.time()

                                # #Put the data through CPC
                                # cpc_hidden = cpc_model.init_hidden(inputs.shape[0], 256, True)
                                # context, hidden = cpc_model.predict(inputs.transpose(1,2), cpc_hidden)

                                # input_sizes_np = to_np(input_sizes)
                                # input_sizes_np = (np.floor_divide(input_sizes_np, 160))
                                # input_sizes = torch.from_numpy(input_sizes_np).int()

                                # Put the CPC output through the DeepSpeech
                                # input_lengths = torch.IntTensor(context.shape[1])
                                if args.cpc_model_path is not None:
                                    context = wav2vec.predict(inputs.transpose(1, 2))
                                    out = model(context)
                                else:
                                    out = model(inputs.transpose(1, 2))

                                float_out = out.float()

                                output_sizes = torch.full((len(inputs),), out.shape[2], dtype=torch.long)

                                neural_net_time = time.time()

                                loss = criterion(float_out.transpose(1, 2).transpose(0, 1).cpu(), targets.cpu(),
                                                 output_sizes.cpu(), target_sizes.cpu())
                                loss_value = loss.item()
                                # print("Validation CTC Loss for {}/{}: {}".format(i, len(val_loader),loss_value))
                                total_loss += loss_value
                                # print("Out shape: {}".format(out.shape))
                                # print("Output sizes: {}".format(output_sizes))

                                decoded_output, _ = decoder.decode(float_out.transpose(1, 2), output_sizes)
                                if True:
                                    target_strings = decoder.convert_to_strings_target(split_targets)

                                decode_time = time.time()
                                # DEBUG CODE
                                # _, max_probs = torch.max(out.transpose(1,2), 2)
                                # # print("Split targets: {}".format(split_targets))
                                # # print("Target strings: {}".format(target_strings))
                                # #print("Max probs shape: {}".format(max_probs.shape))
                                # for i in range(len(out.transpose(1,2))):
                                # 	print("Max probs at i: {}".format(max_probs[i].data.cpu().numpy()))
                                # 	print("Decoded output {}: {}".format(i, decoder.int_to_str(max_probs[i].data.cpu().numpy())))
                                # 	print("Target string: {}: {}".format(i, decoder.int_to_str(split_targets[i].numpy())))
                                wer, cer = 0, 0
                                for x in range(len(target_strings)):
                                    if (args.use_beamsearch):
                                        transcript, reference = decoded_output[x], target_strings[x][0]
                                    else:
                                        transcript, reference = decoded_output[x][0], target_strings[x][0]
                                    # if (x == 0):
                                    # 	print("Predicted: {}".format(transcript))
                                    # 	print("Reference: {}".format(reference))
                                    # wer += decoder.wer(transcript, reference) / float(len(reference.split()))
                                    # cer += decoder.cer(transcript, reference) / float(len(reference))
                                    wer += decoder.wer(transcript, reference)
                                    cer += decoder.cer(transcript, reference)

                                wer_time = time.time()
                                total_wer += wer
                                total_cer += cer
                                del out
                                avg_data_time += data_time - end
                                avg_decode_time += decode_time - neural_net_time
                                avg_wer_time += wer_time - decode_time
                                # if i % 10 == 0:
                                # 	print("{}/{}: Data: {:.3f}({:3f})\tDecode: {:3f}({:3f})\tWER Time: {:3f}({:3f})".format(
                                # 		i, len(test_sampler), data_time - end, avg_data_time/(i+1),
                                # 		decode_time - neural_net_time, avg_decode_time/(i+1),
                                # 		wer_time - neural_net_time, avg_wer_time/(i+1)))
                                end = time.time()
                            wer = total_wer / len(test_loader.dataset)
                            cer = total_cer / len(test_loader.dataset)
                            total_loss = total_loss / len(test_loader)
                            wer *= 100
                            cer *= 100
                            print("Params: Alpha: {:.2f}--Beta: {:.2f}--Beam size: {:.2f}".format(lm_alpha, lm_beta,
                                                                                                  beam_width))
                            print("RESULT:\tAverage CTC Loss: {:.3f}\tAverage WER: {:.3f}\tAverage CER {:.3f}".format(
                                total_loss, wer, cer))
                            if wer < best_wer:
                                best_params = {'alpha': lm_alpha, 'beta': lm_beta, 'beam_size': beam_width}
                                best_wer = wer
                                print("Best WER: {:.3f} CER: {:3f} -- Params: {}".format(wer, cer, best_params))
                    # print("TEST RESULT: \tAverage CTC Loss: {:.3f}\tAverage WER: {:.3f}\tAverage CER {:.3f}".format(total_loss, wer, cer))
