# Speech Emotion Augmentation using CycleGAN
## Overview
This repository contains the code and resources for my Bachelor's thesis project focusing on using Generative Adversarial Networks (GANs), particularly CycleGAN, for speech emotion augmentation. The project aims to transfer different emotions within the speech domain, allowing for the transformation of emotional characteristics in audio samples.

## Dataset
The project utilizes the Speech Emotion Recognition (SER) dataset, comprising audio recordings from 10 speakers in each of the two languages included(English and Chinese). The audio files are pre-processed into spectrograms and further quantized using the Fourier series to prepare the data for the emotion transfer process.

## Methodology
The core methodology involves the implementation of CycleGAN, a type of GAN known for its ability to perform unpaired image-to-image translation, adapted for the transfer of emotional characteristics in speech signals. The CycleGAN framework facilitates the transformation between different emotional states in the audio domain.

## File Structure
### Evaluation/: Contains the FID measurement and an improvement on the performance of the SER model.
### Generative_Model/: Includes the implementation of the CycleGan model and training for converting the source data to N target classes.
### Mel_Spectograms/: Includes converting audio files(.wav) to spectograms.
### Report/: The pdf file of my bachelor's thesis.
### Speech-Emotion-Recognition-models/: Includes training ResNet-50 on the dataset.
## Requirements
Python 3.x
PyTorch (or any other deep learning framework)
Librosa (for audio processing)

## References
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Any-to-many Voice Conversion Using Inverse Short-time Fourier Transform for Faster Conversion](https://arxiv.org/abs/2302.08296)
