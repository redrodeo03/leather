# Leather Quality Classifier Web Application

This is a web application that uses a trained InceptionV3 model to classify leather quality. The model can classify leather images into 4 different categories.

## Features

- Modern, responsive web interface
- Drag and drop image upload
- Real-time image preview
- Classification results with confidence scores
- Support for common image formats (JPG, PNG)

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Flask
- Other dependencies listed in requirements.txt

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd leather-classifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Make sure you have the model file:
- The model file `inceptionNetV3_100eVAL_16b_v2_model.h5` should be in the root directory
- This file contains the trained InceptionV3 model for leather classification

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Click the upload area or drag and drop an image file
2. Preview the image
3. Click "Classify Leather" to get the prediction
4. View the classification result and confidence score

## Model Information

- Base Model: InceptionV3
- Input Size: 224x224 pixels
- Classes: 4 leather quality categories
- Training Accuracy: >97%
- Validation Accuracy: >97%

## Technical Details

- Frontend: HTML5, CSS3, JavaScript
- Backend: Flask (Python)
- Deep Learning: TensorFlow/Keras
- UI Framework: Bootstrap 5

## License

[Your License Information] 