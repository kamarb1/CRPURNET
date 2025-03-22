
# **README: Reproducibility and Documentation**
# CRPURNET

## **Title:** Enhanced Chained Residual U-Net for Precise Segmentation of Thyroid Nodules in Ultrasound Images
**Published in:** *The Visual Computer*  

## **Overview**  
This repository contains the implementation of our proposed methodology as described in the paper *Enhanced Chained Residual U-Net for Precise Segmentation of Thyroid Nodules in Ultrasound Images*, published in *The Visual Computer*. The code facilitates the replication of our experiments, allowing researchers to evaluate the results presented in the manuscript.

## **Installation**  
Ensure you have Python 3.10.13 installed. Then, install the necessary dependencies using:  
```bash
pip install -r requirements.txt
```
If using Conda, create a virtual environment and install dependencies:  
```bash
conda create --name myenv python=3.10.13  
conda activate myenv  
pip install -r requirements.txt  
```

## **Dependencies & Requirements**  
The project requires the following libraries:  
- **Python** (3.10.13)  
- **TensorFlow** (2.10.1)  
- **NumPy** (1.26.4)  
- **OpenCV** (4.9.0.80)  
- **scikit-learn** (1.4.1.post1)
- **scikit-image** (0.24.0)
- **Matplotlib** (3.8.3)  

Ensure all dependencies are installed via `pip install -r requirements.txt` or manually install them using `pip install <package_name>`.

## **Usage Guide**  

### **Step 1: Clone the Repository**  
```bash
git clone https://github.com/kamarb1/CRPURNET.git
cd your-repo-folder
```

### **Step 2: Data Preprocessing**  
Run the preprocessing script before training the model:
```bash
python data_preprocessing.py
```
After running the script, you will have:
- `X_train.npy`, `y_train.npy`
- `X_valid.npy`, `y_valid.npy`
- `X_test.npy`, `y_test.npy`

These files can then be used for training the segmentation model.

### **Step 3: Train the Model**  
Run the training script using the default configuration or specify a config file:  
```bash
python Train.py 
```


### **Step 4: Model Evaluation**  
After training, evaluate the model on the test dataset:  
```bash
python evaluate.py 
```


## **Description of Key Algorithms**  
1. **Metrics**  
   To evaluate the performance of the segmentation model, the following metrics were used:

- **Dice Coefficient**:  
  Measures the overlap between the predicted mask and the ground truth.
  A higher Dice score (closer to 1) indicates better segmentation.

- **Jaccard Index (IoU)**:  
  Another overlap metric, also known as Intersection over Union (IoU) 
  It is slightly more sensitive to small segmentation errors than the Dice coefficient.

- **Precision**:  
  Calculates the proportion of correctly predicted positive pixels among all predicted positives
  A higher precision indicates fewer false positives.

- **Recall**:  
  Measures how many actual positive pixels were correctly identified
  This metric is important when missing a positive region could have critical consequences.

Each metric provides unique insights into the model’s performance, helping to balance false positives and false negatives while ensuring accurate segmentation.

2. ** Loss Function:**  
   - The loss function used is **Binary Cross Entropy (BCE)** for better segmentation performance.   
   - It is commonly used for binary segmentation tasks as it optimizes pixel-wise classification.  

## **Data Availability**  

The datasets used in this project can be downloaded from the following links:

- **TN3K Dataset**: [Download Here](https://drive.google.com/file/d/1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F/view?usp=sharing)
- **DDTI Dataset**: [Download Here](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images)
  


## **Reproducibility**  
To ensure reproducibility, we provide:  
- Complete code and dataset access  
- Step-by-step instructions for execution  
- Fixed random seeds for consistency (`numpy.random.seed(42)`)  

## **Citation**  
If you use this repository, please cite our paper as follows:  
```bibtex
@article{
  author = {bouhdiba kamar, meddeber lila, meddeber mohamed, zouagui tarik},
  title = {Enhanced Chained Residual U-Net for Precise Segmentation of Thyroid Nodules in Ultrasound Images},
  journal = {The Visual Computer},
  year = {2025},
  doi = {DOI}
}
```

## **Contact**  
For questions, please contact:  
📧 Email: kamar.bouhdiba@univ-usto.dz

