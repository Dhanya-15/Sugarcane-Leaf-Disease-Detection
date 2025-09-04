# Sugarcane Leaf Disease Detection 🌱  

This project focuses on **automatic detection and classification of sugarcane leaf diseases** using deep learning techniques. The system uses **transfer learning with ResNet18, EfficientNet, GRU, and CBAM attention modules**, combined with a **sugarcane verifier** model to reject non-sugarcane images. The model is explainable with **LIME visualizations** and achieves an accuracy of **97.8%** on the test dataset.  


## 📊 Dataset
- Dataset source: [Mendeley Data – Sugarcane Leaf Disease Dataset](https://data.mendeley.com/datasets/9twjtv92vk/1)  
- It is a **labeled dataset** with the following classes:  
  - **Main Classes:** Healthy, Dried, Diseased  
  - **Subclasses of Diseased:** Pokka Boeng, Brownspot, Viral Disease, Banded Chlorosis, Grassy Shoot, Settrot, Smut  
- Data augmentation was applied to balance the dataset, resulting in **~3300 images per class**.  

## ⚙️ Features
- **Deep Learning Models:** ResNet18, EfficientNet-B1 with CBAM, GRU Classifier.  
- **Sugarcane Verifier:** Rejects non-sugarcane leaves before classification.  
- **Performance Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix.  
- **Explainability:** LIME visualizations highlight important regions influencing predictions.  
- **High Accuracy:** Achieved **97.8% classification accuracy**.  

## 🚀 Installation & Setup
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/sugarcane-leaf-disease-detection.git
   cd sugarcane-leaf-disease-detection
   
2. Install dependencies
   pip install torch torchvision timm scikit-learn seaborn matplotlib tqdm lime
   
3. Run the code
   Training - python train.py
   Testing and evaluation - python test.py
   LIME (Explainabilty) - python explain.py --image path_to_image.jpg

## 📈 Results
- Overall Accuracy: 97.8%  
- Evaluation Metrics:
  - Precision: 0.97  
  - Recall: 0.98  
  - F1-Score: 0.98  
- Confusion Matrix: Demonstrates strong class-wise performance across all categories (Healthy, Dried, and Diseased subclasses).  
- Explainability: LIME visualizations clearly highlight the diseased regions of sugarcane leaves, improving interpretability of predictions.  
- Robust Verification: Non-sugarcane images are filtered out by the sugarcane verifier before disease classification.  

## 🔮 Future Work
- Deploy the model as a mobile or web-based application to support farmers in real-time disease detection.  
- Integrate the system with IoT devices and drones for field-level monitoring of crops.  
- Extend the dataset to include more disease varieties and environmental conditions for improved generalization.  
- Explore lightweight architectures (e.g., MobileNet, EfficientNet-Lite) for faster inference on low-power devices.  
- Add multi-language support in the application to make it farmer-friendly across regions.  

