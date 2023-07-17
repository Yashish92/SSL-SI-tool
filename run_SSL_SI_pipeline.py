"""
Author : Yashish Maduwantha

Runs the SSL_SI models on a given .wav audio file

1. Run feature extractor to extract given SSL acoustic features after necessary pre processing
2. Run the pretrained SI system by feeding the extracted SSL features to estimate and save the TVs

"""
import os
from run_saved_model import run_model
import argparse
from feature_extract import feature_extract

def get_parser():
    """
    :Description: Returns a parsers with custom arguments cli
    :return: parser
    """
    parser = argparse.ArgumentParser(description='Run the SI pipeline',
                                     epilog="do data processing, extract huBERT feats and evaluate on the saved SI model")
    parser.add_argument('-m', '--model', type=str, default='xrmb',
                        help='set which SI system to run, xrmb trained (xrmb) or hprc trained (hprc)')
    parser.add_argument('-f', '--feats', type=str, default='hubert',
                        help='set which SSL pretrained model to be used to extract features')
    parser.add_argument('-i', '--path', type=str, default='sample_audio',
                        help='path to directory with audio files')
    parser.add_argument('-o', '--out_format', type=str, default='mat',
                        help='output TV file format (mat or npy)')
    return parser

def main():
    # use arg parser
    args = get_parser().parse_args()

    SI_model = args.model
    feats = args.feats
    audio_dir = args.path
    out_format = args.out_format

    for file in os.listdir(audio_dir):
        if file.endswith('.wav'):
            audio_file = audio_dir + '/' + file
            file_name = file[:-4]
            # Run feature extractor script to extract SSL features
            f_extractor = feature_extract(audio_file, feats)  # create feature extractor instance
            feature_data, no_segs, audio_len = f_extractor.run_extraction()

            # Load and run saved SSL_SI model and generate final TV outputs
            run_model(feature_data, file_name, audio_len, SI_model, out_format=out_format)

            print("TVs extracted for " + file)

    print("TV Extraction Completed")

if __name__ == '__main__':
    main()











