🌍 Satellite Image Classification using CNN (EuroSAT Dataset)

📌 Project Overview
This project implements a Convolutional Neural Network (CNN) to classify satellite images from the EuroSAT dataset.
We restricted the dataset to 3 classes:

🌲 Forest

🏠 Residential

🌊 SeaLake

The goal was to build a deep learning pipeline that supports sustainable development applications such as land monitoring, urban planning, and environmental protection.

🎯 Key Objectives

Preprocess satellite images and apply data augmentation

Train a CNN model from scratch for multi-class classification

Evaluate performance using accuracy, confusion matrix, and classification report

Connect the project with United Nations Sustainable Development Goals (SDGs)

🛠️ Tech Stack

Python 🐍

TensorFlow / Keras (Deep Learning)

Scikit-learn (Evaluation metrics)

Matplotlib & Seaborn (Visualization)

Pandas, NumPy (Data handling)

📂 Project Structure
├── eurosat_filtered/        # Filtered dataset (Forest, Residential, SeaLake)
├── image_dataset.csv        # CSV with image paths & labels
├── Modelenv.v1.h5           # Trained CNN model
├── class_indices.json       # Mapping of classes
├── notebook.ipynb           # Jupyter Notebook / Colab file
└── README.md                # Project documentation

🚀 Implementation
1️⃣ Data Preparation

Filtered dataset into 3 classes

Applied ImageDataGenerator with augmentation:

Rotation (45°)

Zoom (0.2)

Horizontal & Vertical flips

Rescaling to 0–1

2️⃣ Model Architecture (CNN)
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(255, 255, 3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))


Optimizer: Adam

Loss: Categorical Crossentropy

Metrics: Accuracy

3️⃣ Results

✅ Accuracy: ~98% on test dataset (9000 images)

✅ High precision & recall across all 3 classes

✅ Clear confusion matrix separation

📊 Evaluation

Classification Report:

              precision    recall  f1-score   support
      Forest       0.97      0.98      0.98      3000
 Residential       1.00      1.00      1.00      3000
     SeaLake       0.98      0.97      0.98      3000


Confusion Matrix:


🌱 Relevance to SDGs

SDG 11: Sustainable Cities & Communities → Monitoring urban areas

SDG 15: Life on Land → Preserving forests

SDG 14: Life Below Water (indirect) → Tracking lakes & water bodies

🔮 Future Work

Implement U-Net for flood detection & segmentation

Apply Transfer Learning (ResNet, VGG16) for improved performance

Extend to all 10 EuroSAT classes

🙌 Acknowledgments

Dataset: EuroSAT on Kaggle

Libraries: TensorFlow, Scikit-learn, Matplotlib, Seaborn
