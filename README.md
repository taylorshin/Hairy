# Hairy

## Abstract
Hairy is a convolutional neural network based on the YOLO network that detects hair follicles in high resolution biomedical images.

## Requirements
- Python 3

Install dependencies of this project.
```
pip install -r requirements.txt
```

## Setup
- The dataset is not provided in this repository
- There should be a *data* folder in the root directory that holds patient data and labels
- There should also be an *out* folder in the root directory that contains the generated logs, plots, and models

## Data
- If you want to add more data, add each new set of patient data in its own folder (i.e. K_data) in the *data* directory
- Add the corresponding label files to the *data/labels/* folder
- Add the paths to the folders/files to the train function in the *train.py* file

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
