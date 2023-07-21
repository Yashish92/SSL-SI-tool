"""
Author : Yashish

Project : SSL-SI with huBERT for XRMB dataset

Description : This script extracts the HuBERT features for given speech audio files

"""
import librosa
import os
import numpy as np
import speechbrain as sb
import torch.nn.functional as F
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
import torch
import torch.nn as nn

# from pandas import DataFrame
import multiprocessing as mp
#
# plt.switch_backend('agg')

BASE_DIR = os.getcwd()
# Set the path for input audio files from the XRMB dataset
fd = '/home/yashish/Academics/Yashish Personal/Research/Nasal_TVs_project/BLSTM_SI_XRMB/data/wav_16k_usable-20220131T051209Z-001/wav_16k_usable/'

# f = open(BASE_DIR + '/lists/valence.txt', 'r')
# lf = list(f)

lf = os.listdir(fd)

# set the directory path for saving the features
dd = BASE_DIR + '/data/XRMB/hubert_feats/'
# dd = BASE_DIR + '/child_feats/'
if not os.path.exists(dd):
    os.makedirs(dd)

count = len(os.listdir(fd))

# HuggingFace model hub
model_hub_hubert = "facebook/hubert-large-ll60k"

model_hubert = HuggingFaceWav2Vec2(model_hub_hubert, save_path='')
# print(model_hubert)

MAXLEN = 2   # seconds
TIMESTEPS = 100 # hubert sampled at 50Hz

def pfile(files):
    fname =files
    fn = files.split('/')[-1]
    # sr = 16000

    # # read audio files using speechbrain
    # source = sb.dataio.dataio.read_audio(fname).squeeze()

    source_ar, sr = librosa.load(fname, sr=16000)  # downsampling to 16Khz

    if len(source_ar) >= MAXLEN * sr:
        source_ext = source_ar[range(MAXLEN * sr)]
        source_ext = torch.from_numpy(source_ext)  # convert to a tensor
    else:
        source_ext = np.pad(source_ar,(0,MAXLEN*sr - len(source_ar)), 'constant', constant_values=0)
        source_ext = torch.from_numpy(source_ext)
        # source_ext = F.pad(source,(1,1), "constant", 0)
        # pad_amt = (sr*MAXLEN)-len(source_ar)
        # source_ext = F.pad(source,[0, pad_amt, 0, 0])
        # source_ext = m(source)
        # print(source_ext.shape)
        # print('in the else case')
        # n_repeats = (MAXLEN * sr) // len(y)
        # y_ed1 = np.tile(y, n_repeats)
        # y_ed = np.concatenate((y_ed1, y[range(MAXLEN * sr - len(y_ed1))]))

    source = source_ext.unsqueeze(0)
    # print(source.shape)

    fea_hubert = model_hubert(source)
    # print(fea_hubert.shape)
    fea_hubert = fea_hubert.squeeze(0)
    # print(fea_hubert.shape)
    fea_hubert_t = torch.transpose(fea_hubert, 0, 1)
    # print(fea_hubert_t.shape)

    # melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=int(0.02 * sr), hop_length=int(0.01 * sr), n_mels=40)
    # # melspec = librosa.feature.melspectrogram(y=y, sr=sr)
    # melspec_db = librosa.power_to_db(melspec, ref=np.max)
    #
    # melspec_db = melspec_db[:, 0:TIMESTEPS]

    # plt.figure()
    # librosa.display.specshow(melspec_db, hop_length=int(0.01 * sr), sr=sr)
    # plt.colorbar()
    # plt.show()

    # pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    # pylab.close()

    torch.save(fea_hubert_t, dd + fn[:-4] + '.pt')

# uncomment to debug without parallelization
for f in lf:
    if f.endswith('.wav'):
        f_path = fd+f
        pfile(f_path)
