"""
Author : Yashish

Project : SSL-SI with huBERT for XRMB dataset

Description : This script generates the training, dev and test files for training the SI system

Notes :
1. Set the paths for directories with HuBERT feats and TVs in appropriate locations
2. Use No_TVs = 9 if only the TV files are available with source features from APP detector


"""
import numpy as np
import os
from scipy.io import loadmat
import torch
# from tensorflow.keras.preprocessing.sequence import pad_sequences

def min_max_norm(X):
    # X_samples = X.shape[0]
    # X_pitch = X.flatten()
    min_val = np.min(X)
    max_val = np.max(X)
    X_pitch = (X - min_val) / (max_val - min_val)
    #print(std)
    return X_pitch



def load_sub_files():

    with open('file_lists/train_subj_list.txt') as f:
        lines = f.readlines()
        for sub in lines:
            train_subjs.append(sub)

    with open('file_lists/test_subj_list.txt') as f:
        lines = f.readlines()
        for sub in lines:
            test_subjs.append(sub)

    with open('file_lists/dev_subj_list.txt') as f:
        lines = f.readlines()
        for sub in lines:
            val_subjs.append(sub)


# def write_file_lists():
#     for file in os.listdir('data/MFCC_feats/'):
#         sub_name = file.split('_')[1] + '\n'  # to match with file list with subject names
#
#         if sub_name in train_subjs[:]:
#             with open('file_lists/train_file_list.txt', 'a') as the_file:
#                 the_file.write(file[:-4])
#                 the_file.write('\n')
#         elif sub_name in val_subjs[:]:
#             with open('file_lists/val_file_list.txt', 'a') as the_file:
#                 the_file.write(file[:-4])
#                 the_file.write('\n')
#         elif sub_name in test_subjs[:]:
#             with open('file_lists/test_file_list.txt', 'a') as the_file:
#                 the_file.write(file[:-4])
#                 the_file.write('\n')

def write_train_files():
    first_train = True
    first_test = True
    first_val = True

    # set the path for directory with huBERT feats
    for file in os.listdir('data/XRMB/hubert_feats'):
        # filter out audio copies in the folder
        if file[-5] != ')' and file[-10] != ')':
            sub_name = file.split('_')[1] + '\n'  # to match with file list with subject names
            hubert_data = torch.load('data/XRMB/hubert_feats/' + file)

            # transform to ndarray
            hubert_data = hubert_data.detach().numpy()
            hubert_data = np.transpose(hubert_data)
            hubert_data = np.pad(hubert_data, pad_width=((0, 1), (0, 0)), mode='edge')

            hubert_data = np.expand_dims(hubert_data, axis=0)

            split_list = file.split('_')

            # check for augmented files
            if len(split_list) > 6:
                file = file[:-8]
            else:
                file = file[:-3]

            # # remove mfcc 0
            # mfcc_data = mfcc_data[1:, :]

            # ## pading using preprocessing function
            # mfcc_data = pad_sequences(mfcc_data, maxlen= time_steps, padding="post", dtype='float32')

            # mfcc_data = mfcc_data.reshape(mfcc_data.shape[1], mfcc_data.shape[0])

            # hubert_data = np.transpose(hubert_data)
            #
            # # perform utterance wise normalization
            # hubert_data = (hubert_data - np.mean(hubert_data, axis=0)) / (np.std(hubert_data, axis=0) + 1e-6)

            # # padding in both edges
            # if mfcc_data.shape[0] < time_steps:
            #     pad_amt_l = (time_steps - mfcc_data.shape[0]) // 2
            #     pad_amt_r = time_steps - (mfcc_data.shape[0] + pad_amt_l)
            #     mfcc_data = np.pad(mfcc_data, ((pad_amt_l, pad_amt_r), (0, 0)), 'constant', constant_values=(0, 0))
            # else:
            #     mfcc_data = mfcc_data[0:time_steps, :]

            # # padding in right edge
            # if hubert_data.shape[0] < time_steps:
            #     pad_amt = time_steps - hubert_data.shape[0]
            #     hubert_data = np.pad(hubert_data, ((0, pad_amt), (0, 0)), 'constant', constant_values=(0, 0))
            # else:
            #     hubert_data = hubert_data[0:time_steps, :]
            #
            # hubert_data = np.expand_dims(hubert_data, axis=0)

            # set directory paths for TV files
            if No_TVs == 6:
                # load TV files
                TV_all_data = loadmat('data/XRMB/XRMB_TV_usable_spknorm-20220131T051304Z-001/XRMB_TV_usable_spknorm/' + file + '_tv.mat')
                TV_data = TV_all_data['tv_norm']
                TV_data = np.transpose(TV_data)
            elif No_TVs == 9:
                TV_all_data = loadmat(
                    'data/XRMB/XRMB_extended_ap_per_pitch/' + file + '_tv.mat')
                TV_data = TV_all_data['tv_norm']
                TV_data = np.transpose(TV_data)

                # normalizing pitch to 0-1 range
                TV_data[:, 8] = min_max_norm(TV_data[:, 8])

            # ## pading using preprocessing function
            # TV_data = pad_sequences(TV_data, maxlen= time_steps, padding="post", dtype='float32')

            # TV_data = TV_data.reshape(TV_data.shape[1], TV_data.shape[0])

            ## padding in both edges
            # if TV_data.shape[0] < time_steps:
            #     pad_amt_l = (time_steps - TV_data.shape[0]) // 2
            #     pad_amt_r = time_steps - (TV_data.shape[0] + pad_amt_l)
            #     TV_data = np.pad(TV_data, ((pad_amt_l, pad_amt_r), (0, 0)), 'constant', constant_values=(0, 0))
            # else:
            #     TV_data = TV_data[0:time_steps, :]

            # padding in right edge
            if TV_data.shape[0] < time_steps:
                pad_amt = time_steps - TV_data.shape[0]
                TV_data = np.pad(TV_data, ((0, pad_amt), (0, 0)), 'constant', constant_values=(0, 0))
            else:
                TV_data = TV_data[0:time_steps, :]

            TV_data = np.expand_dims(TV_data, axis=0)

            if sub_name in train_subjs[:]:
                if first_train:
                    x_train = hubert_data
                    y_train = TV_data
                    first_train = False
                else:
                    x_train = np.vstack((x_train, hubert_data))
                    y_train = np.vstack((y_train, TV_data))
            elif sub_name in val_subjs[:]:
                if first_val:
                    x_val = hubert_data
                    y_val = TV_data
                    first_val = False
                else:
                    x_val = np.vstack((x_val, hubert_data))
                    y_val = np.vstack((y_val, TV_data))
            elif sub_name in test_subjs[:]:
                if first_test:
                    x_test = hubert_data
                    y_test = TV_data
                    first_test = False
                else:
                    x_test = np.vstack((x_test, hubert_data))
                    y_test = np.vstack((y_test, TV_data))

    if No_TVs ==6:
        np.save('data/Train_files/x_train_200_hubert_postpad.npy', x_train)
        np.save('data/Train_files/x_val_200_hubert_postpad.npy', x_val)
        np.save('data/Train_files/x_test_200_hubert_postpad.npy', x_test)

        np.save('data/Train_files/y_train_200_hubert_postpad.npy', y_train)
        np.save('data/Train_files/y_val_200_hubert_postpad.npy', y_val)
        np.save('data/Train_files/y_test_200_hubert_postpad.npy', y_test)
    elif No_TVs ==9:
        np.save('data/Train_files/x_train_200_hubert_ap_per_pitch.npy', x_train)
        np.save('data/Train_files/x_val_200_hubert_ap_per_pitch.npy', x_val)
        np.save('data/Train_files/x_test_200_hubert_ap_per_pitch.npy', x_test)

        np.save('data/Train_files/y_train_200_hubert_ap_per_pitch.npy', y_train)
        np.save('data/Train_files/y_val_200_hubert_ap_per_pitch.npy', y_val)
        np.save('data/Train_files/y_test_200_hubert_ap_per_pitch.npy', y_test)

    print("Done")

if __name__ == '__main__':
    train_subjs = []
    test_subjs = []
    val_subjs = []

    # Need to be fine tuned
    No_TVs = 6
    time_steps = 200
    # max_timesteps = 0
    # min_timesteps = 2000
    tot = 0
    # count = 0

    load_sub_files()
    # write_file_lists()
    write_train_files()



    

