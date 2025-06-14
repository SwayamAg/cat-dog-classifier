# 🐱🐶 Cat vs Dog Image Classifier

This project uses a Convolutional Neural Network (CNN) to classify images as either **cats** or **dogs**. The model was trained using TensorFlow and Keras, and achieves a validation accuracy of **82.6%**.

## 📁 Project Structure

```
CAT_DOG_CNN/
├── Dataset/                   # (Optional) Folder for image dataset
├── cat_dog_cnn_model.h5       # Trained model (tracked via Git LFS)
├── download.jpg               # Sample test image 1
├── download (1).jpg           # Sample test image 2
├── train_model.ipynb          # Notebook to train the model
├── Prediction.ipynb           # Notebook to test model predictions
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignored files (e.g., .ipynb_checkpoints)
├── .gitattributes             # Git LFS config (e.g., *.h5)
└── README.md                  # Project documentation
```

---

## 📌 Features

- Built with **TensorFlow** and **Keras**
- Uses **Convolutional Neural Network (CNN)** with:
  - Conv2D, BatchNormalization, MaxPooling2D layers
- Model architecture includes:
  - Multiple Conv-BN-MaxPooling blocks
  - ~11 million parameters
- Final accuracy: **82.6%**

---

## 🧠 Model Architecture (Simplified)

```
Input → Conv2D(32) → BN → MaxPool 
      → Conv2D(64) → BN → MaxPool 
      → Conv2D(128) → BN → ...
      → Dense → Output (Binary)
```

Total Parameters: **11.7M**  
Trainable Parameters: **11.7M**  
Non-trainable: **448**

---

## 🖼️ Predicting a New Image

You can load and predict an image using `Prediction.ipynb`.  

### Sample Code:
```python
from keras.preprocessing import image
import numpy as np
from tensorflow import keras

# Load model
model = keras.models.load_model('cat_dog_cnn_model.h5')

# Load image
img_path = 'download.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
label = 'Dog' if prediction[0][0] > 0.5 else 'Cat'

print(f"Prediction: {label}")
```

---

## ✅ Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## 📈 Accuracy

| Metric             | Value     |
|--------------------|-----------|
| Validation Accuracy| **82.6%** |

---

## 🚀 Future Improvements

- Add more training data for better generalization
- Use data augmentation to reduce overfitting
- Try transfer learning (e.g., MobileNetV2, ResNet50)
- **Streamlit UI for demo (coming soon!)**
- Deploy and share public demo link

---

## 💡 Note on Git LFS

> **Important:**  
> This project uses **Git Large File Storage (LFS)** to track the model file (`cat_dog_cnn_model.h5`).  
> Please install and initialize Git LFS before cloning or pulling this repo:
>
> ```bash
> git lfs install
> git clone https://github.com/SwayamAg/cat-dog-classifier.git
> ```

---

## 🤝 Contributions

Open for improvements and suggestions. Feel free to fork and enhance the project!

---

## 📸 Example Output

Prediction on `download.jpg`:
```
Prediction: Dog
```

---

## 🧾 License

This project is open-source and available under the MIT License.
