# RNA-DNA background
This is a repo for background interaction analysis code used in "Modelling Background Level in RNA-DNA Interaction Data with Machine Learning Methods" article.

# Notebooks 
- `background_binning.ipynb` - binning mRNA trans-interaction for background profile creation.
- `data_preparation.ipynb` - generation features to predict background from.
- `unet_learn.ipynb` - training 1D U-Net for predicting background in 256Kb genomic windows.
- `mlp_learn.ipynb` - training simple 2 layer MLP model for predicting background in separate bins.

# RNADNA_background
Model classes and functions for data preparation, training, evaluation and visualization. 

# Models 
`models` folder contains `.pth` weights for single U-Net and several MLP models (used primarely in the current work) trained on GRID-seq and RADICL-seq data with full or selected set of features. 
