# Bright2Nuc

Here you can find the structure of the code as well as some relevant information.

## Dependencies

For running the code, you need to have Python 3.6 or higher installed. In addition, these are the main dependencies:

```json
{
   "cudatoolkit":"10.1.243",
   "cudnn":"7.6.5",
   "h5py":"2.10",
   "hdf5":"1.10.6",
   "imageio":"2.6.1",
   "keras":"2.2.4",
   "matplotlib":"3.1.1",
   "numpy":"1.18.5",
   "python":"3.6.10",
   "scikit-image":"0.15.0",
   "scikit-learn":"0.21.3",
   "scipy":"1.3.0",
   "tensorflow":"1.14.0",
   "tensorflow-gpu":"1.14.0",
   "pandas":"1.0.3"
}
```

## How to use the code

The pipeline is based on folders. Please put your data manually in the corresponding folders as illustrated by the figure above.  
The folder names must not be changed as this will stop the pipeline from functioning.

The config.json file in the config folder must be used to set parameters for the pipeline. After this the pipeline is started by executing the main.py file.  
Predictins from the testset will be saved after training to the Results folder which will automatically be created.

From there evaluations using jupyter-notebooks from the Evaluation folder can be used for visual and statistical assessment.  
Therefore only the path in the jupyter-notebook files has to be adapted.

Possible setting in the config.json file are: 
```json
"path": "/.../", #Set the path to your project directory here
"use_pretrained_weights": false, # Set to true if you want to use pretrained weights e.g. "/bright2nuc/pretrained_model/Bright2NucWeights.hdf5"
"pretrained_weights": false, # Set a relative file path from your project directory with the filename here. 
"batchsize": 2, # Set the batchsize depeding on your GPU capabilities
"iterations_over_dataset": 200, # Set how many iterations over the dataset should be taken for learning. 
It might stop automatically if no improvement on the validation set was measured after 25 epochs

Set data augmentation parameters here
"save_augmented_images": false, # true or false
"resample_images": false, # true or false
"std_normalization": false, # true or false
"feature_scaling": false, # true or false
"horizontal_flip": false, # true or false
"vertical_flip": false, # true or false
"poission_noise": false, # false or float
"rotation_range": false, # false or float (degrees)
"zoom_range": false, # false or float (magnification)
"contrast_range": false, # false or float
"brightness_range": false, # false or float
"gamma_shift": false, # false or float (gamma shift parameter)
"threshold_background_image": false, # true or false
"threshold_background_groundtruth": false, # true or false
"gaussian_blur_image": false, # true or float
"gaussian_blur_label": false, # true or  # true or false
"binarize_mask": false # true or false

"loss_function": "mse",
"image_size": false, # false or tuple with dimensions of desired image size in format [x-dim, y-dim, (z-dim), channels],
e.g. [128,128,3]
"seeds", false # true or false
"calculate_uncertainty": false # true or false
"evaluation": true # true or false
```

For running the code, you can simply run it using:

```bash
python main.py --config ./config.json
```

## Training steps:

1. Import of data in all common image file formats that are processable with scikit-image (e.g. .jpg, .tiff, .png) and .npy files.
2. Initialization of a model with pre-trained weights or random weights.
3. Split of ‘train’-folder into 80% training and 20% validation set and randomly shuffles the training data.
4. Normalization of data to the range between 0 and 1 based on the train dataset’s minimum and maximum pixel value. The data will be re-normalized when saved after testing.
5. Batch creation and data augmentation on the fly.
6. Training for the set epoch length using early stopping of training if the validation accuracy has not improved for the last epochs. Using the Adam optimizer (Kingma and Ba, 2014).
7. Real time monitoring of training with Tensorboard or terminal.
8. Saving of the best model during training.
9. Automated evaluation of the trained model on the test data and saving of predicted labels.
10. For pixel-wise regression Bright2Nuc will calculate the uncertainty for each image.

11. Therefore Monte Carlo dropout is used to evaluate 20 different models. The uncertainty estimation is saved to the project directory.

12. Experiment settings are automatically saved to a logbook in order to simplify experiment monitoring.

| Parameter | Explanation of the .json parameter|
| ------ | ------ |
| Path | Relative path to project directory containing the data, pretrained models, results and evaluation |
| Path to pre-trained weights | Relative path to pre-trained weights |
| Batch size | Set integer, it depends on your GPU / CPU capabilities |
| Number of iterations over dataset | Integer, using early stopping if no improvement is measured on the validation set after 25 epochs |
| Data augmentation | Resample images (yes/no), std-mean-normalization (yes/no), feature scaling (yes/no), vertical and horizontal flip (yes/no), zoom (factor), rotation (degrees), add Poisson noise (amound), zoom range (factor), contrast (factor), brightness (factor), gamma shift (factor), threshold background of image (yes/no), threshold background of ground truth (yes/no), Gaussian blur of image (factor), Gaussian blur of groundtruth (factor), binarize groundtruth (yes/no), save augmented images and masks to folder (yes/no) |
| Loss function | Choose from the following: MSE, MAE, dice loss, binary cross entropy, categorical cross entropy, binary crossentropy dice, malis loss |
| Number of classes | Set to the number of classes in your dataset|
| Seeds | Random seeds can be set for reproducibility of experiments |
| Calculate uncertainty | Uncertainty can be calculated using MC dropout |
| Evaluation | Based on the task the model will automatically calculate relevant metrics for a quantitative evaluation and sample images for a qualitative evaluation and save them to the 'evaluation' and 'insights' folders which are automatically created |

## Run examples:
One example is in the examples folder.

To run them only the corresponding config.json file from the example folder must be copied into the configs folder and renamed to "config.json".

Then the main.py can be executed and the example will run with the provided data. 

Please dont expect to achieve competitive results on these datasets, as they are very small and only for illustration purposes.
