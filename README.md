# Detectify-CICIDS2017
Your documentation is already clear and informative! Here's a lightly refined version with improved flow, consistency, and formatting while keeping all technical and instructional details intact:

---

# CICIDS 2017 Repository

## 📊 Overview
This repository contains a collection of Jupyter notebooks crafted to analyze the **CICIDS 2017** dataset—one of the most comprehensive datasets for intrusion detection system (IDS) research. The notebooks guide you through end-to-end workflows including data exploration, preprocessing, and training of machine learning models.

## 🚀 Features

- **Automated Dataset Download**: Easily retrieves the CICIDS 2017 dataset.
- **Exploratory Data Analysis (EDA)**: Visualizes and summarizes key patterns in the data.
- **Model Training**:
  - **Binary Classification**: Logistic Regression, Support Vector Machine.
  - **Multi-Class Classification**: K-Nearest Neighbors, Decision Tree, Random Forest.
  - **Deep Learning Models**: Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), Deep Neural Network (DNN), applied to both binary and multi-class tasks.

## 🧰 Usage
1. Clone this repository.
2. Open the notebooks in your Jupyter environment.
3. Follow the inline instructions to run the code and analyze the results.

## ⚙️ Setting Up the Conda Environment

Set up your Conda environment using the following steps:

```bash
# 1. Create a new environment
conda create -n cicids python=3.9

# 2. Activate the environment
conda activate cicids

# 3. Install core dependencies
pip install numpy pandas seaborn matplotlib scikit-learn tensorflow

# 4. Install additional tools
pip install missingno imbalanced-learn wget

# 5. Install Jupyter Notebook
pip install jupyter notebook

# 6. Set up the IPython kernel for Jupyter
pip install ipykernel

# 7. Register the environment with Jupyter
python -m ipykernel install --user --name=cicids
```

## 📦 Requirements
Ensure the following Python packages are installed:

- `numpy`, `pandas`, `seaborn`, `matplotlib`
- `missingno`, `imbalanced-learn`, `scikit-learn`
- `tensorflow` (or `keras` for deep learning models)
- `wget` for dataset downloads

## 📂 Dataset Details

The CICIDS 2017 dataset is available in three formats:
- Raw network traffic (PCAPs)
- Labeled NetFlow data
- Machine Learning-ready CSV files

In this project, we use the **MachineLearningCSV.zip** version. During download, the file is renamed to **MachineLearningCVE.zip** to match the filename required by the `MachineLearningCSV.md5` checksum.

You can find more information at the [CICIDS 2017 official dataset page](https://www.unb.ca/cic/datasets/ids-2017.html).

## 📚 References

1. [CICIDS 2017 Machine Learning Repository](https://github.com/djh-sudo/CICIDS2017-Machine-Learning/blob/main/README.md)  
2. [Data Preprocessing Notebook](https://github.com/liangyihuai/CICIDS2017_data_processing/blob/master/data_preprocessing_liang.ipynb)  
3. [DNN and Preprocessing Repository](https://github.com/fabian692/DNN-and-preprocessing-cicids-2017)  
4. [Intrusion Detection Notebook](https://github.com/noushinpervez/Intrusion-Detection-CICIDS2017/blob/main/Intrusion-Detection-CIC-IDS2017.ipynb)  
5. [CICIDS 2017 ML Preprocessing](https://github.com/mahendradata/cicids2017-ml)  
6. [Autoencoder Model](https://github.com/fasial634/Autoencoder-model-for-CICIDS-2017-/blob/main/Autoencoder.ipynb)  
7. [Data Cleaning and Random Forest](https://github.com/Moetezafif-git/cicids2017)

## 📝 License
This project is licensed under the **MIT License**.

---

Let me know if you’d like to turn this into a `README.md` file, or generate badges and visuals for GitHub!
