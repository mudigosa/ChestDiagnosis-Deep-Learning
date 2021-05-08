# ChestXNet

Chest X-ray Disease Diagnosis by Samarth Keshari, Gaurav Bhatt, Vivek Vishal

## Installation

The software is built using pytohn 3.8.5 and uses following packages

```
focal-loss-torch==0.0.7
pandas==1.2.3
Pillow==8.0.1
scikit-learn==0.23.2
torch==1.7.0
urllib3==1.26.4
```
You can automatically install all of these packages by first cloning the repo https://github.com/samarthkeshari/CS598-ChestXNet.git. Then navigate into the project directory and run `pip install -r requirements.txt`.


## Usage

There are three APIs that you can use.

### Data Acquisition

`cd src`.

The purpose of this API is to download the sample Chest X-ray images from NIH website. This API will download 12 .tar files from NIH website in `../data/` also extract the images from the tar files in `../data/images` 

To run, `python data_acquisition.py`

### Model Training

`cd src`.

The purpose of this API is to start the Model Training. It can be controlled using parameters that are maintained in file `train_params.json`. If required, change the parameters file.

To run, `python model_training.py <modelname ('VGG16' or 'VGG16-ATTN')> <custom parameters json file(optional)>`. The trained model weights will be saved aftr each epoch in `./models` or the path specified in custom parameters file as `<epoch#-modelname.pth>`. The model metrics `loss, roc-auc` will get saved in `./metrics` or the path specified in custom parameters file as `<modelname-TRAIN-LOSS.csv.csv>` and `<modelname-TRAIN-CLASS-AUC.csv>` .

If a validation run is required during training, then set the parameter `"perform_validation":"True" `in the custom parameters json file. The validation metrics will get saved in `./metrics` or the path specified in custom parameters file as `<modelname-VAL-LOSS.csv.csv>` and `<modelname-VAL-CLASS-AUC.csv>` .

Note: Runinng training and validation together will take lot of time, so it is advised to run validation seperately using the `Model Testing API` as described below


### Model Testing

`cd src`.
The purpose of this API is to start the Model Validation. It can be controlled using parameters that are maintained in file `test_params.json`. If required, change the parameters file.

To run, `python model_testing.py <modelname <trained-model-filename> <custom parameters json file(optional)>`. The validation metrics will get saved in `./metrics` or the path specified in custom parameters file as `<modelname-VAL-LOSS.csv.csv>` and `<modelname-VAL-CLASS-AUC.csv>` .

## Important Point
- The dataset size is about 45 GB, so training and vaidation must be done using GPU. It took around 6 hours to train the model with 1 GPU.
- The code supports training using only 1 GPU
