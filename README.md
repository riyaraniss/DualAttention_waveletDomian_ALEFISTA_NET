# Wavelet-Domain ALE–FISTA-Net with Dual Attention for LOFAR Enhancement

This repository provides a reference implementation of a wavelet-domain FISTA-Net with dual attention (channel and spatial) and Adaptive Line Enhancer (ALE) preprocessing, designed for LOFAR spectrum enhancement in underwater passive sonar applications. The codebase supports both training and inference using a unified script and includes dataset handling utilities for experiments conducted on the ShipsEar dataset.

The implementation accompanies the paper: Wavelet-Domain Dual-Attention FISTA-Net for Noise-Resilient LOFAR Spectrum Enhancement (under review)


📌 Key Features

* ALE preprocessing for tonal enhancement prior to learning

* Wavelet-domain sparse representation for noise-robust reconstruction

* FISTA-Net with learnable shrinkage thresholds

* Dual attention mechanism

    * Channel attention for adaptive subband weighting

    * Spatial attention for salient time–frequency localization

* Unified training and inference pipeline

* Compatible with self-supervised and synthetic ground-truth training

* Reproducible experiments with pinned software versions
