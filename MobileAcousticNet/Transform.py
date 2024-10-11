import math, random
import torch
import torchaudio
from spafe.features.gfcc import gfcc
from spafe.features.lfcc import lfcc
from spafe.features.pncc import pncc
from torchaudio import transforms
# from IPython.display import Audio
from spafe.features.bfcc import bfcc
from spafe.features.mfcc import mfcc
from spafe.utils.preprocessing import SlidingWindow
import time
import numpy as np


class AudioUtil():
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
            return aud
        if (new_channel == 1):
            resig = sig[:1, :]
        else:
            resig = torch.cat([sig, sig])
        return ((resig, sr))

    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud
        if (sr == newsr):
            return aud
        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if (num_channels > 1):
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])
        return ((resig, newsr))

    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms  # 采样率取余1000，表示一个毫秒采集多少个样本，然后乘以预先定义好的最大音频长单（毫秒）

        if (sig_len > max_len):
            sig = sig[:, :max_len]
        elif (sig_len < max_len):
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)
        return (sig, sr)

    @staticmethod
    def spectro_gram(aud, n_mels=13, n_fft=1024, hop_len=512):
        sig, sr = aud

        # lfccs = lfcc(sig, fs=int(sr), pre_emph=True,
        #              pre_emph_coeff=0.97, window=SlidingWindow(0.03, 0.015, "hamming"),
        #              nfilts=128, nfft=2048, low_freq=0, high_freq=int(sr) / 2, normalize="mvn")
        # mfccs = mfcc(sig, fs=int(sr), pre_emph=True,
        #              pre_emph_coeff=0.97, window=SlidingWindow(0.03, 0.015, "hamming"),
        #              nfilts=128, nfft=2048, low_freq=0, high_freq=int(sr) / 2, normalize="mvn")
        # gfccs = gfcc(sig, fs=int(sr), pre_emph=True,
        #              pre_emph_coeff=0.97, window=SlidingWindow(0.03, 0.015, "hamming"),
        #              nfilts=128, nfft=1024, low_freq=0, high_freq=int(sr) / 2, lifter=None,
        #              normalize="mvn")
        # lfccs = torch.from_numpy(lfccs.T)
        # mfccs = torch.from_numpy(mfccs.T)
        # gfccs = torch.from_numpy(gfccs.T)
        #
        # lfccs = torch.unsqueeze(lfccs, dim=0)
        # mfccs = torch.unsqueeze(mfccs, dim=0)
        # gfccs = torch.unsqueeze(gfccs, dim=0)
        #
        # input_feature = torch.concatenate((lfccs, mfccs, gfccs), dim=0)
        # input_feature = input_feature.float()
        MFCC = torchaudio.transforms.MFCC(sample_rate=sr,n_mfcc=13)(sig)
        LFCC = torchaudio.transforms.LFCC(sample_rate=sr,n_lfcc=13)(sig)
        input_feature = torch.concatenate((MFCC,LFCC),dim=0)
        # print(input_feature.shape)
        # input_feature = torchaudio.transforms.MFCC(sample_rate=sr,n_mfcc=13)(sig)

        return input_feature

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
        return aug_spec
