# Clothing Apparel Detection

## Project Overview

This project implements an advanced object detection system specifically designed for identifying and classifying clothing and apparel items in images. It utilizes state-of-the-art deep learning techniques and the TensorFlow Object Detection API to provide accurate and efficient detection results.

## Features

- **Multi-class Detection**: Capable of detecting multiple clothing items and accessories in a single image.
- **High Accuracy**: Utilizes SSD MobileNet V1 COCO model for optimal performance and accuracy.
- **Customizable Label Map**: Easily extendable to include new clothing categories via the `labelmap.pbtxt` file.
- **Visualization Tools**: Includes utilities for visualizing detection results with bounding boxes and labels.
- **Evaluation Metrics**: Implements CorLoc and mAP metrics for model performance evaluation.
- **Flexible Input Processing**: Supports various input formats including TFRecord and XML.
- **Mask Support**: Optional instance segmentation mask support for more detailed item delineation.

## Project Structure
- `apparel_detection.py`: Main script for running the object detection model.
- `labelmap.pbtxt`: Custom label map for defining clothing categories.
- `utils.py`: Utility functions for data processing and visualization.
- `requirements.txt`: List of required Python packages.


## Setup and Installation

1. Ensure Python 3.x is installed.
2. Set up a virtual environment:
   ```bash
   python3 -m venv venv
   ```
   Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
   Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
1. Prepare your dataset and label map.
2. Run the `apparel_detection.py` script with the appropriate command-line arguments.
3. Visualize the detection results using the provided utilities.


## Evaluation

The project includes comprehensive evaluation tools:
- `per_image_evaluation.py`: Computes CorLoc and true/false positives.
- Visualization utilities in `visualization_utils.py` for qualitative assessment.

## Customization

- Modify `config_util.py` to adjust model configurations.
- Update input processing in `dataset_util.py` for custom data formats.

## Contributing

Contributions to improve the project are welcome. Please follow the standard fork-and-pull request workflow.
