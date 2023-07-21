"""
Author : Yashish Maduwantha

This performs acoustic feature extraction from the specified pretrained SSL model

"""

import librosa
import numpy as np
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
import torch

class feature_extract:
    def __init__(self, ssl_model):
        self.MAXLEN = 2  # seconds
        self.TIMESTEPS = 100  # hubert, wavLM sampled at 50Hz

        if ssl_model == 'hubert':
            self.model_hub_hubert = "facebook/hubert-large-ll60k"
            self.model_extractor = HuggingFaceWav2Vec2(self.model_hub_hubert, save_path='')
        elif ssl_model == 'wavlm':
            self.model_hub_wavlm = "microsoft/wavlm-large"
            self.model_extractor = HuggingFaceWav2Vec2(self.model_hub_wavlm, save_path='')

    def run_extraction(self, wave_file):
        source_ar, sr = librosa.load(wave_file, sr=16000)
        audio_len = int(self.MAXLEN * sr)

        file_len = len(source_ar)

        # chunk_num = len(s_wav) // audio_len
        # extra_len = len(s_wav) % audio_len
        # if extra_len > audio_len // 2:
        #     trim_len = (chunk_num + 1) * audio_len
        # else:
        #     trim_len = chunk_num * audio_len

        first_test = True
        if len(source_ar) <= audio_len:
            no_segs = 1
            pad_amt = audio_len - len(source_ar)
            spk_wav = np.concatenate([source_ar, np.zeros(pad_amt, dtype=np.float32)])
            spk_wav_tensor = torch.from_numpy(spk_wav)
            spk_wav_tensor_un = spk_wav_tensor.unsqueeze(0)
            spk_wav_ssl = self.model_extractor(spk_wav_tensor_un)
            spk_wav_ssl = spk_wav_ssl.squeeze(0)

            spk_wav_ssl_npy = spk_wav_ssl.detach().numpy()
            spk_wav_ssl_npy = np.pad(spk_wav_ssl_npy, pad_width=((0, 1), (0, 0)), mode='edge')
            spk_wav_ssl_npy = np.expand_dims(spk_wav_ssl_npy, axis=0)
            # spk_wav = np.expand_dims(spk_wav, axis=0)
        elif len(source_ar) > audio_len:
            # spk_wav = s_wav[:audio_len]
            no_segs = len(source_ar) // audio_len + 1
            for i in range(0, no_segs):
                if first_test:
                    spk_data_seg = source_ar[:audio_len]
                    spk_seg_tensor = torch.from_numpy(spk_data_seg)
                    spk_seg_tensor_un = spk_seg_tensor.unsqueeze(0)
                    spk_seg_ssl = self.model_extractor(spk_seg_tensor_un)

                    spk_seg_ssl = spk_seg_ssl.squeeze(0)
                    spk_seg_ssl_npy = spk_seg_ssl.detach().numpy()
                    spk_seg_ssl_npy = np.pad(spk_seg_ssl_npy, pad_width=((0, 1), (0, 0)), mode='edge')
                    spk_seg_ssl_npy = np.expand_dims(spk_seg_ssl_npy, axis=0)

                    spk_wav_ssl_npy = spk_seg_ssl_npy
                    first_test = False
                elif i < no_segs - 1:
                    spk_data_seg = source_ar[audio_len * i:audio_len * (i + 1)]
                    spk_seg_tensor = torch.from_numpy(spk_data_seg)
                    spk_seg_tensor_un = spk_seg_tensor.unsqueeze(0)
                    spk_seg_ssl = self.model_extractor(spk_seg_tensor_un)

                    spk_seg_ssl = spk_seg_ssl.squeeze(0)
                    spk_seg_ssl_npy = spk_seg_ssl.detach().numpy()
                    spk_seg_ssl_npy = np.pad(spk_seg_ssl_npy, pad_width=((0, 1), (0, 0)), mode='edge')
                    spk_seg_ssl_npy = np.expand_dims(spk_seg_ssl_npy, axis=0)

                    spk_wav_ssl_npy = np.vstack((spk_wav_ssl_npy, spk_seg_ssl_npy))
                elif i == no_segs - 1:
                    pad_amt = (audio_len * no_segs) - file_len
                    spk_data_seg = np.concatenate([source_ar[(audio_len * i):], np.zeros(pad_amt, dtype=np.float32)])
                    spk_seg_tensor = torch.from_numpy(spk_data_seg)
                    spk_seg_tensor_un = spk_seg_tensor.unsqueeze(0)
                    spk_seg_ssl = self.model_extractor(spk_seg_tensor_un)

                    spk_seg_ssl = spk_seg_ssl.squeeze(0)
                    spk_seg_ssl_npy = spk_seg_ssl.detach().numpy()
                    spk_seg_ssl_npy = np.pad(spk_seg_ssl_npy, pad_width=((0, 1), (0, 0)), mode='edge')
                    spk_seg_ssl_npy = np.expand_dims(spk_seg_ssl_npy, axis=0)

                    spk_wav_ssl_npy = np.vstack((spk_wav_ssl_npy, spk_seg_ssl_npy))

        # print('feature extraction done')

        return spk_wav_ssl_npy, no_segs, file_len


if __name__ == '__main__':
    f_extractor = feature_extract('test_audio/spk1_snt1.wav','hubert')
    data, segs, aud_len = f_extractor.run_extraction()








