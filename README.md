# PointNet: Point Cloud Classification and Segmentation

This repository contains an implementation of **PointNet** for point cloud
classification and segmentation, based on the original papers and datasets:

## References

- C. R. Qi, H. Su, K. Mo, and L. J. Guibas,  
  *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation*,  
  arXiv:1612.00593, 2017.  
  https://arxiv.org/pdf/1612.00593

- Z. Wu, S. Song, A. Khosla, X. Tang, and J. Xiao,  
  *3D ShapeNets: A Deep Representation for Volumetric Shapes*,  
  arXiv:1406.5670 [cs.CV], 2014.

- L. Yi, V. G. Kim, D. Ceylan, I. Shen, M. Yan, H. Su, A. Lu, Q. Huang,  
  A. Sheffer, and L. Guibas,  
  *A Scalable Active Framework for Region Annotation in 3D Shape Collections*,  
  ACM Transactions on Graphics (TOG), vol. 35, no. 6, article 210, 2016.

## Overview

The project focuses primarily on **classification**, with an additional
implementation of the **segmentation architecture**.

- Classification is trained and evaluated on **ModelNet10**
- Segmentation is implemented but **not trained or evaluated**
  (training would require **ShapeNet**, which was not included in this project)

## Implementation Details

- Implemented in **PyTorch**, using `torch.nn` modules
- Custom low-level layers were avoided to reduce implementation complexity
- The **T-Net** module was implemented as a separate neural network component  
  (inspired by the implementation by Mason McGough:  
  https://gist.github.com/Mason-McGough)

## Results

- Best classification model achieved **92.73% test accuracy** on ModelNet10
- The `results/` directory contains:
  - Saved model weights (`.pt`)
  - Training and testing accuracy/loss plots
  - Confusion matrix of the best-performing model

## Notes

- A standalone `test.py` script was not fully developed due to time constraints
- Available documentation is written in **Croatian**, as this project was
  submitted as a university mini-project

