# SSL-SI-tool

* Author: Yashish Maduwantha
* Email: yashish@terpmail.umd.edu

## Description
This repository holds two Acoustic-to-Articulatory Speech Inversion (SI) systems trained on the Wisconsin XRMB dataset and the HPRC dataset respectively. The model architecture and training are based on the papers [Audio Data Augmentation for Acoustic-to-articulatory Speech Inversion](https://arxiv.org/abs/2205.13086) and ["Acoustic-to-articulatory Speech Inversion with Multi-task Learning"](https://www.isca-speech.org/archive/pdfs/interspeech_2022/siriwardena22_interspeech.pdf). The pretrained SI systems in this repository have been trained with self-supervised based features (HuBERT and wavLM) as acoustic inputs compared to the 13 MFCCs used in the papers above.  

1. Model trained on XRMB dataset : Estimates 6 TVs
2. Model trained on HPRC dataset : Trained with a MTL framework and estimates 9 TVs + Source features (Aperiodicity, Periodicity and Pitch)

Check the two papers above to refer to more information on the types of TVs estimated by each model. 

## Installation Guide
The SI systems were trained in a conda environment with Python 3.8.13 and tensorflow==2.10.0. The HuBERT pretrained models used to extract acoustic features have been trained in PyTorch.

1. Installation method 1:

First install tensorflow and we recommend doing that in Conda following the steps [here](https://www.tensorflow.org/install/pip).

We also use a number of off the shelf libraries which are listed in [requirements.txt](requirements.txt). Follow the steps below to install them.

```bash
$ pip install speechbrain
$ pip install librosa
$ pip install transformers
```

2. Installation method 2 : Installing inidividual libraries from the [requirements.txt](requirements.txt) file.
```bash
$ pip install -r requirements.txt
```

We recommed following method 1 since it will automatically take care of compatible libraries incase there have been new realase versions of respective libraries.

Note : If you run the SI system on GPUs to extract TVs (recommended for lareger datasets), make sure the cuDNN versions for pyTorch (installed by speechbrain) and the one installed with Tensorflow are compatible.

## Run SI tool pipeline

Execute [run_SSL_SI_pipeline.py](run_SSL_SI_pipeline.py) script 
to run the SI pipeline which performs the following 'steps',

1. Run [feature_extract.py](feature_extract.py) script to do audio segmentation and extract specified SSL features using the [speechbrain](https://github.com/speechbrain/speechbrain/) library
2. Load the pre-trained SSL-SI model and evaluate on the extracted SSL feature data generated in step 1 
3. Save the predicted Tract Variables (TVs)

### Python command line usage:
```bash
usage: run_SSL_SI_pipeline.py [-h] [-m MODEL] [-f FEATS] [-i PATH]
                              [-o OUT_FORMAT]

Run the SI pipeline

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        set which SI system to run, xrmb trained (xrmb) or
                        hprc trained (hprc)
  -f FEATS, --feats FEATS
                        set which SSL pretrained model to be used to extract
                        features, hubert to use HuBERT-large and wavlm to use
                        wavLM-large pretrained models
  -i PATH, --path PATH  path to directory with audio files
  -o OUT_FORMAT, --out_format OUT_FORMAT
                        output TV file format (mat or npy)

```

### Example for running the ML pipeline

1. Run the pipeline from end to end (executes all 3 steps)
```python
python run_SSL_SI_pipeline.py -m xrmb -f hubert -i test_audio/ -o 'mat'
```

## Note

The SI systems trained with wavLM features will be added in the future. Only set -f parameter to 'hubert' at this point to run the models. 

## License
This project is licensed under the MIT License - see the LICENSE file for details
