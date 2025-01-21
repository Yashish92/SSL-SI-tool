"""
Author : Yashish Maduwantha

Run the pretrained SI system by feeding the extracted SSL features to estimate and save the TVs

"""

import numpy as np
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input, Dropout, Embedding, Masking, TimeDistributed, BatchNormalization, ReLU, GRU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from KalmanSmoother import kalmansmooth
from scipy.io import savemat
import feature_extracter

import os
import datetime
from datetime import date

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def run_model(input_features, file_name, audio_len, SI_model, out_format='mat'):
    # base_dir = './'
    out_path = "TV_output_" + str(date.today()) + "_H_" + str(datetime.datetime.now().hour) + '/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    fs = 16000
    # Load saved model
    if SI_model == 'xrmb':
        saved_model = 'saved_models/best_XRMB_model/GRU_model_XRMB_hubert_utterance-wise_original_mfcc_2023-07-21_H_14/net/BLSTM_model.h5'
    elif SI_model == 'hprc':
        saved_model = 'saved_models/best_hprc_model/multi_GRU_model_EMA_IEEE_hubert_utterance-wise_original_mfcc_2023-07-09_H_2/net/GRU_model_multi_task.h5'

    loaded_model = load_model(saved_model)

    y_predict = loaded_model.predict(input_features, verbose=0)

    # since MTL model also outputs phoneme labels
    if SI_model == 'hprc':
        y_predict = y_predict[0]

    for i in range(0, y_predict.shape[0]):
        seg_TVs = y_predict[i]
        if i == 0:
            final_TVs = seg_TVs
        else:
            final_TVs = np.concatenate((final_TVs, seg_TVs), axis=0)

    ## remove padded zeros at the end based on audio length
    audio_time = audio_len / fs
    TV_len = int(audio_time * 100)  # TV sampling rate is 100Hz

    final_TVs = np.transpose(final_TVs)

    final_TVs = final_TVs[:, 0:TV_len]

    # run kalman smoother
    tv_smth = kalmansmooth(final_TVs)

    ## save the final concatenated TVs to a given output_format
    if out_format == 'mat':
        # save to a .mat file
        mdic = {"tv": tv_smth}
        savemat(out_path + file_name + '_' + SI_model +'_tv_predict.mat', mdic)
    elif out_format == 'npy':
        # save to a .npy file
        np.save(out_path + file_name + '_' + SI_model + '_tv_predict.npy', tv_smth)


if __name__ == '__main__':
    f_extractor = feature_extracter.feature_extract('test_audio/spk1_snt1.wav', 'hubert')
    data, segs, aud_len = f_extractor.run_extraction()

    run_model(data, 'spk1_snt1.wav', aud_len, 'hprc', out_format='mat')
