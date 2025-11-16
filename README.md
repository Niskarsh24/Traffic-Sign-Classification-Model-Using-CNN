# Traffic Sign Classification Model Using CNN

A deep learning project that uses Convolutional Neural Networks (CNN) to classify Indian traffic signs. This project includes a trained model and two different user interface options (Streamlit and Gradio) for easy deployment and testing.

## Project Overview

This project implements a CNN-based image classification system trained on the **Indian Traffic Sign Classification dataset** from Hugging Face. The model can identify **85 different types of traffic signs** including speed limits, prohibitory signs, warning signs, and mandatory signs.

## Features

- **High Accuracy Classification**: Achieves ~85% validation accuracy on Indian traffic signs
- **Custom CNN Architecture**: Designed specifically for traffic sign recognition
- **Dual UI Options**: 
  - Streamlit app for a clean, simple interface
  - Gradio app for quick testing and sharing
- **Pre-trained Model**: Includes `indian_traffic_sign_cnn.h5` trained on Google Colab with GPU
- **85 Traffic Sign Classes**: Comprehensive coverage of Indian traffic signs

## Requirements

```
tensorflow>=2.10.0
streamlit
gradio
pillow
numpy
pandas
matplotlib
scikit-learn
datasets
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Niskarsh24/Traffic-Sign-Classification-Model-Using-CNN.git
cd Traffic-Sign-Classification-Model-Using-CNN
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the trained model file `indian_traffic_sign_cnn.h5` is in the project directory.

## Dataset

This project uses the **Indian Traffic Sign Classification** dataset from Hugging Face:
- **Source**: `kannanwisen/Indian-Traffic-Sign-Classification`
- **Total Images**: 5,726 images
- **Training Set**: 4,580 images (80%)
- **Validation Set**: 1,146 images (20%)
- **Number of Classes**: 85 different traffic sign categories
- **Image Size**: Resized to 64x64 pixels for training

### Traffic Sign Classes

The model can classify 85 different types of traffic signs including:
- **Speed Limits**: 5, 15, 20, 30, 40, 50, 60, 70, 80 km/h
- **Prohibitory Signs**: No Entry, No Parking, U-Turn Prohibited, etc.
- **Warning Signs**: Dangerous Dip, Falling Rocks, Slippery Road, etc.
- **Mandatory Signs**: Keep Left, Turn Right, Compulsory Sound Horn, etc.
- **Informational Signs**: School Ahead, Pedestrian Crossing, etc.

## Model Architecture

The CNN architecture consists of:

```python
model = tf.keras.Sequential([
    # First Convolutional Block
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    
    # Second Convolutional Block
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    # Third Convolutional Block
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    # Dense Layers
    Flatten(),
    Dense(256, activation='relu'),
    Dense(85, activation='softmax')  # 85 classes
])
```

**Training Configuration**:
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Epochs**: 10
- **Batch Size**: 32
- **Image Preprocessing**: Resized to 64x64, normalized to [0,1]

##  Model Performance

**Training Results** (10 epochs):
- **Final Training Accuracy**: ~98.77%
- **Final Validation Accuracy**: ~84.90%
- **Final Training Loss**: 0.0413
- **Final Validation Loss**: 0.8094

**Performance by Epoch**:
| Epoch | Training Acc | Validation Acc | Training Loss | Validation Loss |
|-------|--------------|----------------|---------------|-----------------|
| 1     | 13.14%       | 39.35%         | 3.7446        | 2.1681          |
| 5     | 87.48%       | 80.63%         | 0.4160        | 0.7634          |
| 10    | 98.77%       | 84.90%         | 0.0413        | 0.8094          |

## Usage

### Option 1: Streamlit App

Run the Streamlit interface:
```bash
streamlit run app.py
```

The Streamlit app provides:
- Simple file upload interface
- Image preview
- Classification results with predicted class name

### Option 2: Gradio App

Run the Gradio interface:
```bash
python gradio_app.py
```

The Gradio app provides:
- Drag-and-drop or click to upload
- Instant predictions
- Shareable link for remote access

### Programmatic Usage

```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model("indian_traffic_sign_cnn.h5")

# Load and preprocess image
image = Image.open("path/to/traffic_sign.jpg")
image = image.convert("RGB")
image = tf.image.convert_image_dtype(tf.constant(np.array(image)), tf.float32)
image = tf.image.resize(image, (64, 64))
image = tf.expand_dims(image, axis=0)

# Make prediction
predictions = model.predict(image)
score = tf.nn.softmax(predictions[0])
predicted_class = np.argmax(score)
```

## 🔍 Training the Model

The model was trained in Google Colab with GPU acceleration. To retrain:

1. Open `TrafficSignClassification.ipynb` in Google Colab
2. Ensure GPU runtime is enabled (Runtime → Change runtime type → GPU)
3. Run all cells sequentially
4. The trained model will be saved as `indian_traffic_sign_cnn.h5`

**Training Process**:
1. Load dataset from Hugging Face
2. Split into 80/20 train/validation
3. Preprocess images (resize to 64x64, normalize)
4. Train CNN for 10 epochs
5. Visualize accuracy and loss curves
6. Save trained model

## Project Structure

```
Traffic-Sign-Classification-Model-Using-CNN/
├── TrafficSignClassification.ipynb  # Training notebook (Colab)
├── app.py                            # Streamlit UI application
├── gradio_app.py                     # Gradio UI application
├── indian_traffic_sign_cnn.h5       # Pre-trained model (15.6 MB)
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

## Use Cases

- **Driver Assistance Systems**: Real-time traffic sign recognition
- **Autonomous Vehicles**: Navigation and compliance systems
- **Educational Tools**: Teaching traffic rules and sign recognition
- **Traffic Management**: Automated sign inventory and monitoring
- **Mobile Applications**: Sign recognition apps for learner drivers

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Troubleshooting

**Model loading error**:
- Ensure `indian_traffic_sign_cnn.h5` is in the same directory
- Check TensorFlow version compatibility

**Image prediction issues**:
- Ensure image is clear and traffic sign is visible
- Model works best with images similar to training data (Indian traffic signs)

**Performance issues**:
- The model may show overfitting (high training accuracy vs validation accuracy)
- Consider adding dropout layers or data augmentation for improvement

## Author

**Niskarsh**
- GitHub: [@Niskarsh24](https://github.com/Niskarsh24)
- email: niskarsh7607@gmail.com

## Acknowledgments

- **Dataset**: Indian Traffic Sign Classification by kannanwisen on Hugging Face
- **Framework**: TensorFlow/Keras
- **Training Platform**: Google Colab with GPU support
- **UI Frameworks**: Streamlit and Gradio for easy deployment

## Contact

For questions or feedback, please open an issue in the GitHub repository.

## Future Enhancements

- [ ] Add data augmentation to improve generalization
- [ ] Implement dropout layers to reduce overfitting
- [ ] Deploy as web service (Flask/FastAPI)
- [ ] Add real-time video processing capability
- [ ] Create mobile app version
- [ ] Expand to other countries' traffic sign datasets
- [ ] Implement transfer learning using a pretrained model to improve accuracy and reduce training time.
