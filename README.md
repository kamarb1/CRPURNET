

Title: [Enhanced Chained Residual U-Net for Precise Segmentation of Thyroid Nodules in Ultrasound Images]

# **README: Reproducibility and Documentation**
# CRPURNET

## **Title:** [Enhanced Chained Residual U-Net for Precise Segmentation of Thyroid Nodules in Ultrasound Images]  
**Published in:** *The Visual Computer*  

## **Overview**  
This repository contains the implementation of our proposed methodology as described in the paper *Enhanced Chained Residual U-Net for Precise Segmentation of Thyroid Nodules in Ultrasound Images*, published in *The Visual Computer*. The code facilitates the replication of our experiments, allowing researchers to evaluate the results presented in the manuscript.

## **Installation**  
Ensure you have Python 3.x installed. Then, install the necessary dependencies using:  
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
git clone https://github.com/your-repo-link.git
cd your-repo-folder
```

### **Step 2: Data Preprocessing**  
Convert raw data into the required format and apply preprocessing steps:  
```bash
python preprocess.py --input data/raw/ --output data/processed/
```

### **Step 3: Train the Model**  
Run the training script using the default configuration or specify a config file:  
```bash
python train.py --config config.yaml
```
To train with custom parameters, modify `config.yaml` or pass arguments:  
```bash
python train.py --epochs 100 --batch_size 32 --learning_rate 0.001
```

### **Step 4: Model Evaluation**  
After training, evaluate the model on the test dataset:  
```bash
python evaluate.py --model model.pth --test_data data/test/
```

### **Step 5: Inference on New Data**  
To test the trained model on new images, run:  
```bash
python inference.py --model model.pth --input new_image.jpg --output result.jpg
```

## **Description of Key Algorithms**  
1. **Feature Extraction Module:**  
   - Implements [methodology used], which extracts important features from images.  
   - Uses convolutional layers with [specific architecture] to enhance key information.  

2. **Custom Loss Function:**  
   - A hybrid loss function combining [e.g., Dice Loss + Cross-Entropy] for better segmentation/classification performance.  

3. **Post-processing with Morphological Operations:**  
   - Applies dilation and erosion techniques to refine the output segmentation.  

## **Data Availability**  

The datasets used in this project can be downloaded from the following links:

- **TN3K Dataset**: [Download Here](https://drive.google.com/file/d/1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F/view?usp=sharing)
- **DDTI Dataset**: [Download Here](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images)
- 


## **Reproducibility**  
To ensure reproducibility, we provide:  
- Complete code and dataset access  
- Step-by-step instructions for execution  
- Fixed random seeds for consistency (`numpy.random.seed(42)`)  

## **Citation**  
If you use this repository, please cite our paper as follows:  
```bibtex
@article{YourName2025,
  author = {Your Name, Co-Authors},
  title = {Your Paper Title},
  journal = {The Visual Computer},
  year = {2025},
  doi = {Your DOI}
}
```

## **Contact**  
For questions, please contact:  
📧 Email: kamar.bouhdiba@univ-usto.dz

