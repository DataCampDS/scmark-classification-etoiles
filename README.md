[![build](https://github.com/ramp-kits/scMARK_classification/actions/workflows/testing.yml/badge.svg)](https://github.com/ramp-kits/scMARK_classification/actions/workflows/testing.yml)
# Data camp M2DS: single-cell type classification challenge

Starting kit for the `scMARK_classification` challenge: classification of cell-types with single-cell RNAseq count data.


# Setup
Run the following command in your prefered local repository

`git clone git@github.com:DataCampDS/scmark-classification-etoiles.git`

# Install requirements

```bash
pip install -r requirements.txt
```

# Download OSF data
You can download the public data via the dedicated python script to be called as follow 

```bash
python download_data.py 
```

# Get started with the challenge
Open the jupyter notebook `scMARK_classification_starting_kit.ipynb` and follow the guide !

# Preprocess notebook
1. Install requirements
2. Download data
3. Create train and test set
4. Watch labels
5. Dealing with oversampling
6. PCA on n_components with 2 way: One resampled and one without resampling
7. Visualisation
8. Preprocess with normalisation
9. Create pipeline
10. Fit on train and predict on test 
11. Balanced accuracy and Confusion matrix