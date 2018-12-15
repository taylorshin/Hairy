# Hairy

## Abstract
Hairy is a convolutional neural network based on the YOLO network that attempts to detect hair follicles.

## Requirements
- Python 3

Install dependencies of this project.
```
pip install -r requirements.txt
```

The dataset is not provided in this repository. To train Hairy, you must include data.hdf5 and image_boxes.txt in the root directory.

## Usage
To train a new model, run the following command:
```
python train.py
```

To detect hair follicles in an image, run the following command:
```
python predict.py
```

Use the help command to see CLI arguments:
```
python predict.py --help
```
