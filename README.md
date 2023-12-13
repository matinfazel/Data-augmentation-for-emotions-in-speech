# Speech Emotion Augmentation using CycleGAN
## Overview
This repository contains the code and resources for my Bachelor's thesis project focusing on using Generative Adversarial Networks (GANs), particularly CycleGAN, for speech emotion augmentation. The project aims to transfer different emotions within the speech domain, allowing for the transformation of emotional characteristics in audio samples.

## Dataset
The project utilizes the Speech Emotion Recognition (SER) dataset, comprising audio recordings from 10 speakers in each of the two languages included(English and Chinese). The audio files are pre-processed into spectrograms and further quantized using Fourier series to prepare the data for the emotion transfer process.

## Methodology
The core methodology involves the implementation of CycleGAN, a type of GAN known for its ability to perform unpaired image-to-image translation, adapted for the transfer of emotional characteristics in speech signals. The CycleGAN framework facilitates the transformation between different emotional states in the audio domain.

## File Structure
Spectogram.ipynb: Contains the Python scripts and Jupyter notebooks used for data preprocessing and converting .wav files to spectograms.
SER.ipynb: Includes the implementation of the CycleGan model and training for converting two classes together.
angry2happy/: Contains a few samples of two classes in the dataset.
## Requirements
Python 3.x
PyTorch (or any other deep learning framework)
Librosa (for audio processing)

## Results
The /results directory contains the generated audio samples from the model trained on the SER dataset. Additionally, evaluation metrics and visualizations are available for assessing the quality and effectiveness of emotion transfer.

## References
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Any-to-many Voice Conversion Using Inverse Short-time Fourier Transform for Faster Conversion](https://arxiv.org/abs/2302.08296)
