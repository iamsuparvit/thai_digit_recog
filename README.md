# Thai Handwritten Digit Recognition with CNN (PyTorch)

## 📌 About

This project focuses on recognizing Thai handwritten digits using a Convolutional Neural Network (CNN) implemented in PyTorch. The model is trained on a publicly available Thai handwritten digit dataset and aims to classify digits (0–9) accurately. To increase robustness and real-world usability, the model has been further improved using data augmentation techniques and fine-tuning with hand-drawn digits collected under varied conditions.

## 🛠 Installation

To get started, clone this repository and install the required Python dependencies:

```bash
git clone https://github.com/your-username/thai-digit-recog.git
cd thai-digit-recog
pip install -r requirements.txt
```
You can also set up a conda environment if preferred:
```
conda create -n thai-digit python=3.10
conda activate thai-digit
pip install -r requirements.txt
```

## 🚀 Usage

1. Train the model
```
python train.py
```
You can configure parameters such as number of epochs, batch size, learning rate, and paths inside train.py.

2. Run evaluation
```
python evaluate.py
```
3. Predict from a custom image
```
python predict.py --img_path ./samples/mydigit.png
```

4. Launch web interface (Gradio)
```
python app.py
```

## 📚 Dataset

The model was primarily trained on the Thai Handwritten Digit Dataset, which contains 10 classes of digits (0–9) written by Thai individuals.

Image size: 28x28 grayscale
Format: PNG images, organized into folders by label
Additional real-world samples (digitally drawn on white background) were added to improve generalization
Data Augmentation (applied to training data only):
Random affine transformation (rotation, translation)
Normalization to match input scale
Conversion to grayscale and resizing to 28×28

## 🧠 Model Architecture

The model uses a Convolutional Neural Network (CNN) architecture with the following structure:

### 🔧 Convolutional Layers
- Conv2D (1→16, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool + Dropout
- Conv2D (16→32, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool + Dropout
### 🔗 Fully Connected Layers
- Linear(32 * 7 * 7 → 128) + ReLU + Dropout
- Linear(128 → 10) (10 output classes)
The model includes batch normalization and dropout layers to improve stability and prevent overfitting.

## 📈 Results

After training for xx epochs on the dataset:

Training Accuracy: N/A
Validation Accuracy: N/A
Confusion Matrix: Shows balanced performance across all digit classes
Real-World Test Accuracy: Slightly lower due to variation in handwriting and noise
Visual examples show accurate predictions on most hand-drawn digits when they are centered and clearly written.

## 🧪 Limitations and Future Work

Known Limitations
The model struggles with digits written in unusual styles or placed far off-center.
Predictions from real-world inputs (e.g., drawn with a mouse or on touchscreens) may still be biased toward certain digits (e.g., predicting 6 frequently).
Future Improvements
Expand dataset with more real-world styled digits from diverse writers.
Add better image preprocessing (e.g., centering, thresholding).
Train longer with early stopping and learning rate scheduling.
Deploy to mobile/web with TensorFlow Lite or ONNX for real-time use.

## 🧠 Training Details

Loss Function: CrossEntropyLoss
Optimizer: Adam
Batch Size: 64
Learning Rate: 0.001
Epochs: 50
Dropout: 0.2–0.3 depending on layer
