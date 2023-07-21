"""
Author :Yashish Maduwantha

Model : BiGRNN model for XRMB data with HuBERT features

Note : This script can also be used for another work on data augmentation for SI. Make sure you set the correct strings to run the appropriate model.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.layers import Bidirectional, UpSampling1D, LSTM, GRU, Dense, Input, Dropout, Embedding, Masking, TimeDistributed, BatchNormalization, ReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import tanh
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from Attention import Attention

import datetime
from datetime import date
# from sklearn.preprocessing import StandardScaler
# from tensorflow_addons.optimizers import NovoGrad
# from tensorflow.keras.utils import to_categorical
# from scipy.stats import pearsonr

from utils import correlation_coefficient_loss

from matplotlib import pyplot
pyplot.switch_backend('agg')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def ppmc(x, y):
    xbar = np.nanmean(x)
    ybar = np.nanmean(y)
    num = np.nansum((x - xbar)*(y - ybar))
    den = np.sqrt(np.nansum((x - xbar)**2))*np.sqrt(np.nansum((y - ybar)**2))
    corr = num/den

    return corr

def compute_corr_score(y_predict, y_true, no_TVs):
    No_TVs = no_TVs
    corr_TVs = np.zeros(No_TVs)
    corr_TVs_pc = np.zeros(No_TVs)
    tot_samples = y_true.shape[0]
    for j in range(0, y_true.shape[0]):
        for i in range(0, No_TVs):
            corr_TVs[i] += ppmc(y_predict[j,:,i], y_true[j, :, i])
            # corr, _ = pearsonr(y_predict[j,:,i], y_true[j, :, i])
            # corr_TVs_pc[i] += corr
    # for j in range(0, y_val.shape[0]):
    #     for i in range(0, No_TVs):
    #         corr_TVs[i] += ppmc(y_predict_val[j,:,i], y_val[j, :, i])
    corr_TVs_avg = corr_TVs/tot_samples
    avg_corr_tvs = np.mean(corr_TVs_avg)
    avg_corr_6TVs = np.mean(corr_TVs_avg[0:6])
    # corr_TVs_pc = corr_TVs_pc/tot_samples

    print("Corr_Average_Test_set :", corr_TVs_avg)
    print("Corr_Average_across_all:", avg_corr_tvs)
    print("Corr_Average_across_6TVs:", avg_corr_6TVs)
    # print("Corr_pearson_func :", corr_TVs_pc)

    return corr_TVs_avg, avg_corr_tvs, avg_corr_6TVs    # print("Corr_pearson_func :", corr_TVs_pc)


def scheduler(epoch, lr):
    if epoch < 20 or lr < 1e-5:
        return lr
    elif epoch%5 == 0:
        return lr * tf.math.exp(-0.1)
    else:
        return lr


def blstm_model_small(sample_x):
    """
    Implmentation of the small blstm model for original data
    :param sample_x: Input tensor
    :return: Model object
    """
    print("-------------sample x shape---------------")
    print(sample_x.shape)
    inp = Input(shape=(sample_x.shape))

    masking_layer = Masking()
    masked_embedding = masking_layer(inp)

    # embedding = Embedding(input_dim=sample_x.shape[1], output_dim=sample_x.shape[1], mask_zero=True)
    # masked_output = embedding(inp)

    forward_layer_1 = GRU(256, return_sequences=True, dropout=0.3)
    backward_layer_1 = GRU(100, activation='relu', return_sequences=True, go_backwards=True)
    forward_layer_2 = GRU(128, return_sequences=True, dropout=0.3)
    backward_layer_2 = GRU(100, activation='relu', return_sequences=True, go_backwards=True)
    # forward_layer_3 = LSTM(128, return_sequences=True, dropout=0.3)
    # # forward_layer_4 = LSTM(100, return_sequences=True, dropout=0.5)


    blstm_1 = Bidirectional(forward_layer_1, input_shape=(TIME_STEPS, hubert_dims))(masked_embedding)
    # atten1 = Attention(return_sequences=True)(blstm_1)
    # bnorm_1 = BatchNormalization()(blstm_1)
    # drp_1 = Dropout(DROPOUT_PROB)(blstm_1)
    blstm_2 = Bidirectional(forward_layer_2)(blstm_1)
    atten2 = Attention(return_sequences=True)(blstm_2)
    # blstm_3 = Bidirectional(forward_layer_3)(blstm_2)
    # blstm_4 = Bidirectional(forward_layer_4)(blstm_3)
    # drp_2 = Dropout(DROPOUT_PROB)(blstm_2)
    dense_1 = TimeDistributed(Dense(128, activation='relu'))(atten2)
    up1 = UpSampling1D(size=2)(dense_1)  # added new upsample layer to match hubert time dimensions to TVs
    bnorm_2 = BatchNormalization()(up1)
    drp_3 = Dropout(DROPOUT_PROB)(bnorm_2)
    # dense_2 = TimeDistributed(Dense(64, activation='relu'))(drp_3)
    # bnorm_3 = BatchNormalization()(dense_2)
    # drp_3 = Dropout(DROPOUT_PROB)(bnorm_3)
    # dense_3 = TimeDistributed(Dense(6, activation='linear'))(drp_3)
    # out = dense_3
    dense_2 = TimeDistributed(Dense(No_TVs))(drp_3)
    out = tanh(dense_2)

    return Model(inputs=inp, outputs=out)



def blstm_model_ext(sample_x):
    """
    Implmentation of the extended blstm model for augmented data
    :param sample_x: Input tensor
    :return: Model object
    """
    print("-------------sample x shape---------------")
    print(sample_x.shape)
    inp = Input(shape=(sample_x.shape))

    masking_layer = Masking()
    masked_embedding = masking_layer(inp)

    # embedding = Embedding(input_dim=sample_x.shape[1], output_dim=sample_x.shape[1], mask_zero=True)
    # masked_output = embedding(inp)

    forward_layer_1 = GRU(256, return_sequences=True, dropout=0.3)
    backward_layer_1 = GRU(256, activation='relu', return_sequences=True, go_backwards=True)
    forward_layer_2 = GRU(256, return_sequences=True, dropout=0.3)
    backward_layer_2 = GRU(256, activation='relu', return_sequences=True, go_backwards=True)
    forward_layer_3 = GRU(128, return_sequences=True, dropout=0.3)


    blstm_1 = Bidirectional(forward_layer_1, input_shape=(TIME_STEPS, hubert_dims))(masked_embedding)
    # bnorm_1 = BatchNormalization()(blstm_1)
    # drp_1 = Dropout(DROPOUT_PROB)(blstm_1)
    blstm_2 = Bidirectional(forward_layer_2)(blstm_1)
    blstm_3 = Bidirectional(forward_layer_3)(blstm_2)
    # blstm_4 = Bidirectional(forward_layer_4)(blstm_3)
    # drp_2 = Dropout(DROPOUT_PROB)(blstm_2)
    # dense_1 = TimeDistributed(Dense(128, activation='relu', kernel_initializer =tf.keras.initializers.HeUniform()))(blstm_3)
    dense_1 = TimeDistributed(Dense(128, activation='relu'))(blstm_3)
    bnorm_2 = BatchNormalization()(dense_1)
    drp_3 = Dropout(DROPOUT_PROB)(bnorm_2)
    # dense_2 = TimeDistributed(Dense(64, activation='relu'))(drp_3)
    # bnorm_3 = BatchNormalization()(dense_2)
    # drp_3 = Dropout(DROPOUT_PROB)(bnorm_3)
    # dense_3 = TimeDistributed(Dense(6, activation='linear'))(drp_3)
    # out = dense_3
    # dense_2 = TimeDistributed(Dense(6, kernel_initializer = tf.keras.initializers.GlorotUniform()))(drp_3)
    dense_2 = TimeDistributed(Dense(6))(drp_3)
    out = tanh(dense_2)

    return Model(inputs=inp, outputs=out)


def fine_tune(x_train, y_train, x_test, y_test, x_val, y_val):
    """
    Model training and evaluation
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param x_val:
    :param y_val:
    :return: Test accuracy evaluated on the test data
    """

    if dataset == 'augmented_mfcc' or dataset == 'augmented_mfcc_plus' or dataset == 'augmentedGausSNR_mfcc' or dataset == 'augmentedIR_mfcc':
        classifier = blstm_model_ext(x_train[0])
    elif dataset == 'original_mfcc' or dataset == 'original_mfcc_plus':
        classifier = blstm_model_small(x_train[0])

    # opt = NovoGrad(lr=1e-3, beta_1=0.95, beta_2=0.5)
    earlystop = EarlyStopping(monitor='val_loss', patience=PATIENCE)
    LR_scheduler = LearningRateScheduler(scheduler)
    opt = Adam(lr=LR)
    # corr_loss = correlation_coefficient_loss
    # classifier.compile(optimizer=opt, loss=corr_loss, metrics=["mse"])
    # classifier.compile(optimizer=opt, loss=MeanSquaredError(), metrics=['mse'])
    classifier.compile(optimizer=opt, loss=MeanAbsoluteError(), metrics=['mae'])
    classifier.summary()

    # fit the model
    history = classifier.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                              verbose=1, validation_data=(x_val, y_val), callbacks=[earlystop, LR_scheduler])

    # compute train accuracy
    _,train_MAE= classifier.evaluate(x_train, y_train, verbose=0)
    # test on data
    _,test_MAE = classifier.evaluate(x_test, y_test, verbose=0)

    # save model
    classifier.save(model_dir + 'BLSTM_model.h5')

    y_predict = classifier.predict(x_test, verbose=0)

    mse = MeanSquaredError()
    test_MSE = mse(y_test, y_predict).numpy()

    # y_predict_val = classifier.predict(x_val, verbose=0)
    corr_avg, avg_corr, avg_corr_6TVs  = compute_corr_score(y_predict, y_test, No_TVs)

    # write a .txt with results and model params
    lines = ['Dataset:' + dataset, 'Normalization:' + norm_type ,'Epochs:' + str(EPOCHS), 'Patience:' + str(PATIENCE), 'Batch size:' + str(BATCH_SIZE), 'Corr_Average:' + str(corr_avg), 'Corr_Average_all:' + str(avg_corr),
             'Corr_Average_6TVs:' + str(avg_corr_6TVs),'MSE:' + str(test_MSE), 'MAE:' + str(test_MAE)]
    with open(out_dir +'log.txt', 'w') as f:
        f.write('\n'.join(lines))

    # plot TVs for 5 samples
    for sample in range(0,5):
        for tv in range(0,No_TVs):
            pyplot.plot(y_predict[sample,:,tv], label='predicted')
            pyplot.plot(y_test[sample,:,tv], label='True')
            pyplot.legend()
            pyplot.title('TV Predictions')
            pyplot.savefig(tv_dir + 'sample_' + str(sample + 1) + '_' + 'tv_' + str(tv + 1) + '.png')
            pyplot.close()
            pyplot.show()

    # plots for loss and accuracy change over epochs
    print('Train MAE: %.3f, Test MAE: %.3f' % (train_MAE, test_MAE))
    # plot loss during training
    # pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    # pyplot.subplot(212)
    # pyplot.title('Accuracy')
    # pyplot.plot(history.history['accuracy'], label='train')
    # pyplot.plot(history.history['val_accuracy'], label='test')
    # pyplot.legend()
    pyplot.savefig(loss_dir + 'losses.png')
    pyplot.close()
    pyplot.show()

    return test_MSE


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    No_TVs = 6
    norm_type = 'utterance-wise' #'utterance-wise','speaker-wise'
    dataset = 'original_mfcc' #'augmented_mfcc' #'augmentedGausSNR_mfcc' 'augmentedIR_mfcc'# 'original_mfcc', 'augmented_mfcc_plus', 'original_mfcc_plus'

    if No_TVs == 6:
        out_dir = "./Model_outs/GRU_model_XRMB_hubert_" + norm_type + '_' +dataset + '_' + str(date.today()) + "_H_" + str(datetime.datetime.now().hour) + "/"
    elif No_TVs == 9:
        out_dir = "./Model_outs/GRU_model_XRMB_hubert_SF_" + norm_type + '_' + dataset + '_' + str(
            date.today()) + "_H_" + str(datetime.datetime.now().hour) + "/"

    loss_dir = out_dir + "loss/"
    model_dir = out_dir + 'net/'
    tv_dir = out_dir + 'tvs/'

    # Create sub directory is dont exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create weights directory if don't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    if not os.path.exists(tv_dir):
        os.makedirs(tv_dir)

    # NUM_CLASSES = 30  #depends of the version of the dataset

    # Define Model Parameters
    # test params defined below.
    # params = {'FILTER_SIZE': (11, 29, 1, 1), 'NUM_FILTERS_CONV': (128, 128, 128), 'DROPOUT_PROB1': 0.5}
    # print(params)

    # NUM_FILTERS_CONV1 = params['NUM_FILTERS_CONV'][0]
    # NUM_FILTERS_CONV2 = params['NUM_FILTERS_CONV'][1]
    # NUM_FILTERS_CONV3 = params['NUM_FILTERS_CONV'][2]
    # if dataset == 'augmented_mfcc_plus' or dataset == 'original_mfcc_plus':
    #     hubert_dims = 39  # 13
    # else:
    #     hubert_dims = 13

    hubert_dims = 1024

    LR = 1e-3
    DROPOUT_PROB = 0.3
    EPOCHS = 500
    PATIENCE = 5
    BATCH_SIZE = 64
    TIME_STEPS = 100

    if norm_type == 'utterance-wise':

        if dataset == 'original_mfcc_plus':
            # for MFCCs Delta and Delta delta
            X_tr = np.load('data/Train_files/x_train_200_mfccplus_postpad.npy')
            X_val = np.load('data/Train_files/x_val_200_mfccplus_postpad.npy')
            X_te = np.load('data/Train_files/x_test_200_mfccplus_postpad.npy')

            Y_tr = np.load('data/Train_files/y_train_200_mfccplus_postpad.npy')
            Y_val = np.load('data/Train_files/y_val_200_mfccplus_postpad.npy')
            Y_te = np.load('data/Train_files/y_test_200_mfccplus_postpad.npy')

        # set the paths for files here when using HuBERT features
        elif dataset == 'original_mfcc':
            if No_TVs == 6:
                X_tr = np.load('data/Train_files_SSL/x_train_200_hubert_postpad.npy')

                # to test on clean speech
                X_val = np.load('data/Train_files_SSL/x_val_200_hubert_postpad.npy')
                X_te = np.load('data/Train_files_SSL/x_test_200_hubert_postpad.npy')
                Y_tr = np.load('data/Train_files_SSL/y_train_200_hubert_postpad.npy')

                # to test on clean speech
                Y_val = np.load('data/Train_files_SSL/y_val_200_hubert_postpad.npy')
                Y_te = np.load('data/Train_files_SSL/y_test_200_hubert_postpad.npy')

            elif No_TVs==9:
                X_tr = np.load('data/Train_files_SSL/x_train_200_hubert_ap_per_pitch.npy')

                # to test on clean speech
                X_val = np.load('data/Train_files_SSL/x_val_200_hubert_ap_per_pitch.npy')
                X_te = np.load('data/Train_files_SSL/x_test_200_hubert_ap_per_pitch.npy')
                Y_tr = np.load('data/Train_files_SSL/y_train_200_hubert_ap_per_pitch.npy')

                # to test on clean speech
                Y_val = np.load('data/Train_files_SSL/y_val_200_hubert_ap_per_pitch.npy')
                Y_te = np.load('data/Train_files_SSL/y_test_200_hubert_ap_per_pitch.npy')


        elif dataset == 'augmented_mfcc':
            # For only MFCCs with Augmented dataset
            X_tr = np.load('data/Train_files/x_train_200_mfcc_aug_postpad.npy')
            # X_val = np.load('data/Train_files/x_val_200_mfcc_aug_postpad.npy')
            # X_te = np.load('data/Train_files/x_test_200_mfcc_aug_postpad.npy')

            # # To test on noisy only speech
            # X_val = np.load('data/Train_files/x_val_200_mfcc_aug_only_postpad.npy')
            # X_te = np.load('data/Train_files/x_test_200_mfcc_aug_only_postpad.npy')

            # To test on clean speech
            X_val = np.load('data/Train_files/x_val_200_mfcc_postpad.npy')
            X_te = np.load('data/Train_files/x_test_200_mfcc_postpad.npy')

            Y_tr = np.load('data/Train_files/y_train_200_mfcc_aug_postpad.npy')
            # Y_val = np.load('data/Train_files/y_val_200_mfcc_aug_postpad.npy')
            # Y_te = np.load('data/Train_files/y_test_200_mfcc_aug_postpad.npy')

            # # To test on noisy only speech
            # Y_val = np.load('data/Train_files/y_val_200_mfcc_aug_only_postpad.npy')
            # Y_te = np.load('data/Train_files/y_test_200_mfcc_aug_only_postpad.npy')

            # To test on clean speech
            Y_val = np.load('data/Train_files/y_val_200_mfcc_postpad.npy')
            Y_te = np.load('data/Train_files/y_test_200_mfcc_postpad.npy')

        elif dataset == 'augmented_mfcc_plus':
            # For only MFCCs with Augmented dataset
            X_tr = np.load('data/Train_files/x_train_200_mfccplus_aug_postpad.npy')
            # X_val = np.load('data/Train_files/x_val_200_mfccplus_aug_postpad.npy')
            # X_te = np.load('data/Train_files/x_test_200_mfccplus_aug_postpad.npy')

            # To test on clean speech
            X_val = np.load('data/Train_files/x_val_200_mfccplus_postpad.npy')
            X_te = np.load('data/Train_files/x_test_200_mfccplus_postpad.npy')

            Y_tr = np.load('data/Train_files/y_train_200_mfccplus_aug_postpad.npy')
            # Y_val = np.load('data/Train_files/y_val_200_mfccplus_aug_postpad.npy')
            # Y_te = np.load('data/Train_files/y_test_200_mfccplus_aug_postpad.npy')

            # To test on clean speech
            Y_val = np.load('data/Train_files/y_val_200_mfccplus_postpad.npy')
            Y_te = np.load('data/Train_files/y_test_200_mfccplus_postpad.npy')

        elif dataset == 'augmentedGausSNR_mfcc':
            # For only MFCCs with Augmented dataset
            X_tr = np.load('data/Train_files/x_train_200_mfcc_augGausSNR_postpad.npy')
            # X_val = np.load('data/Train_files/x_val_200_mfcc_augGausSNR_postpad.npy')
            # X_te = np.load('data/Train_files/x_test_200_mfcc_augGausSNR_postpad.npy')

            # To test on noisy only speech
            X_val = np.load('data/Train_files/x_val_200_mfcc_augGausSNR_only_postpad.npy')
            X_te = np.load('data/Train_files/x_test_200_mfcc_augGausSNR_only_postpad.npy')

            # # To test on clean speech
            # X_val = np.load('data/Train_files/x_val_200_mfcc_postpad.npy')
            # X_te = np.load('data/Train_files/x_test_200_mfcc_postpad.npy')

            Y_tr = np.load('data/Train_files/y_train_200_mfcc_augGausSNR_postpad.npy')
            # Y_val = np.load('data/Train_files/y_val_200_mfcc_augGausSNR_postpad.npy')
            # Y_te = np.load('data/Train_files/y_test_200_mfcc_augGausSNR_postpad.npy')

            # To test on noisy only speech
            Y_val = np.load('data/Train_files/y_val_200_mfcc_augGausSNR_only_postpad.npy')
            Y_te = np.load('data/Train_files/y_test_200_mfcc_augGausSNR_only_postpad.npy')

            # # To test on clean speech
            # Y_val = np.load('data/Train_files/y_val_200_mfcc_postpad.npy')
            # Y_te = np.load('data/Train_files/y_test_200_mfcc_postpad.npy')

        elif dataset == 'augmentedIR_mfcc':
            # For only MFCCs with Augmented dataset
            X_tr = np.load('data/Train_files/x_train_200_mfcc_augIR_postpad.npy')
            # X_val = np.load('data/Train_files/x_val_200_mfcc_augGausSNR_postpad.npy')
            # X_te = np.load('data/Train_files/x_test_200_mfcc_augGausSNR_postpad.npy')

            # To test on noisy only speech
            X_val = np.load('data/Train_files/x_val_200_mfcc_augIR_only_postpad.npy')
            X_te = np.load('data/Train_files/x_test_200_mfcc_augIR_only_postpad.npy')

            # # To test on clean speech
            # X_val = np.load('data/Train_files/x_val_200_mfcc_postpad.npy')
            # X_te = np.load('data/Train_files/x_test_200_mfcc_postpad.npy')

            Y_tr = np.load('data/Train_files/y_train_200_mfcc_augIR_postpad.npy')
            # Y_val = np.load('data/Train_files/y_val_200_mfcc_augGausSNR_postpad.npy')
            # Y_te = np.load('data/Train_files/y_test_200_mfcc_augGausSNR_postpad.npy')

            # To test on noisy only speech
            Y_val = np.load('data/Train_files/y_val_200_mfcc_augIR_only_postpad.npy')
            Y_te = np.load('data/Train_files/y_test_200_mfcc_augIR_only_postpad.npy')

            # # To test on clean speech
            # Y_val = np.load('data/Train_files/y_val_200_mfcc_postpad.npy')
            # Y_te = np.load('data/Train_files/y_test_200_mfcc_postpad.npy')

    elif norm_type == 'speaker-wise':
        # For only MFCCs with Speaker-wise normalized data
        X_tr = np.load('data/Train_files/x_train_200_mfcc_postpad_snormalized.npy')
        X_val = np.load('data/Train_files/x_val_200_mfcc_postpad.npy')
        X_te = np.load('data/Train_files/x_test_200_mfcc_postpad.npy')

        Y_tr = np.load('data/Train_files/y_train_200_mfcc_postpad_snormalized.npy')
        Y_val = np.load('data/Train_files/y_val_200_mfcc_postpad.npy')
        Y_te = np.load('data/Train_files/y_test_200_mfcc_postpad.npy')

    accs = fine_tune(X_tr, Y_tr, X_te, Y_te, X_val, Y_val)

    print('Test set MSE:', accs)





