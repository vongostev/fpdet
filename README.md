[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/vongostev/fpdet.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/vongostev/fpdet/context:python)
# fpdet
 Basic analytical functions for a few-photon light detection.

 This module includes:
 - Direct and inverse photodetection matrices for binomial and subbinomial detection models (**d_matrix** and **invd_matrix**)
 - Function to calculate the photocount distribution from the photon-number one (**P2Q**) and vice versa (**Q2P**)
 - Some utility functions like fidelity, moments, convolution and entropy

Requirements: 
- numpy
- scipy
