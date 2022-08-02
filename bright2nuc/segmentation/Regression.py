'''
InstantDL
Written by Dominik Waibel and Ali Boushehri

In this file the functions are started to train and test the networks
'''

from bright2nuc.utils import *
from bright2nuc.segmentation.unet_models import UNetBuilder

class Regression(object):
    def __init__(   self,
                    use_algorithm, # todo: it is not used!
                    path,
                    pretrained_weights = None,
                    batchsize = 2,
                    iterations_over_dataset = 100,
                    data_gen_args = None,
                    loss_function = "mse",
                    num_classes = 1,
                    nuclei_size = 30,
                    image_size = None,
                    seeds=False,  # todo: it is not used!
                    calculate_uncertainty = False,
                    evaluation = False):

        self.use_algorithm = "Regression" # todo: correct this
        self.path = path
        self.pretrained_weights = pretrained_weights
        self.batchsize = batchsize
        self.iterations_over_dataset = iterations_over_dataset
        self.loss_function = loss_function
        self.num_classes = num_classes
        self.image_size = image_size
        self.calculate_uncertainty = calculate_uncertainty
        self.nuclei_size = nuclei_size
        # add a comment what 30 is
        if self.nuclei_size != 30:
            self.resize_factor = 30./nuclei_size
        else:
            self.resize_factor = 1
        
        if data_gen_args is None:
            self.data_gen_args = dict()
        else:
            self.data_gen_args = data_gen_args
        self.evaluation = evaluation
    
    def data_prepration(self): 
        '''
        Get the number of input images and their shape
        If the last image dimension,. which should contain the channel information (1 or 3) is not existing e.g. for 
        (512,512) add a 1 as the channel number.
        '''
        if self.image_size == False or self.image_size == None:
            Training_Input_shapes, num_channels = get_input_image_sizes(self.iterations_over_dataset, self.path)
        else:
            Training_Input_shapes = self.image_size
            num_channels = int(self.image_size[-1])
            data_path = self.path + '/train/'



        if os.path.isdir(self.path + "/train/"):
            number_input_images = len(os.listdir(self.path + "/train/"))
        else:
            number_input_images = 0
        logging.info("Number of input folders is: %s" % number_input_images)

        '''
        Import filenames and split them into train and validation set according to 
        the variable -validation_split = 20%
        '''
        data_path = self.path + '/train/'
        train_image_files, val_image_files = training_validation_data_split(data_path)

        steps_per_epoch = np.sum(Training_Input_shapes, axis = 0)[0]

        self.epochs = self.iterations_over_dataset
        logging.info("Making: %s steps per Epoch" % steps_per_epoch)
        return [Training_Input_shapes, num_channels,
                        data_path, train_image_files, val_image_files, steps_per_epoch]



    def data_generator(self, data_path, training_Input_shapes, num_channels, train_image_files, val_image_files):
        '''
        Prepare data as a Training and Validation set
        Args:
            data_path: Path to folder containing the dataset
            Training_Input_shape: Shape of the input images in the train folder
            num_channels: Number of channels (e.g.: 3 for RGB)
            train_image_files: List of filenames contained in the train set
            val_image_files: List of filenames contained in the validation set
        return:
            Two data generators (train & validation) and the number of channels of the groundtruth (label)
        '''


        img_file_label_name = os.listdir(data_path + "/groundtruth/")[0]
        logging.info("img_file_label_name: %s" % img_file_label_name)
        Training_Input_shape_label = np.shape(np.array(import_image(data_path + "/groundtruth/" + img_file_label_name)))
        num_channels_label = Training_Input_shape_label[-1]
        if all([num_channels_label != 1, num_channels_label != 3]):
            num_channels_label = 1

        TrainingDataGenerator = training_data_generator(training_Input_shapes,
                                                            self.batchsize, num_channels,
                                                            num_channels_label,
                                                            train_image_files,
                                                            self.data_gen_args,
                                                            data_path,
                                                            self.resize_factor)
        ValidationDataGenerator = training_data_generator(training_Input_shapes,
                                                              self.batchsize, num_channels,
                                                              num_channels_label,
                                                              val_image_files,
                                                              self.data_gen_args,
                                                              data_path,
                                                              self.resize_factor)
        return TrainingDataGenerator, ValidationDataGenerator,num_channels_label

    def load_model(self):
        '''
        Build a 2D or 3D U-Net model and initialize it with pretrained or random weights
        Args:
            network_input_size: Dimensions of one input image (e.g. 128,128,3)
            num_channels_label: Number of channels of the groundtruth (e.g.: 3 for RGB)
        returns:
            A 2D or 3D UNet model
        '''
        if self.pretrained_weights == False:
            self.pretrained_weights = None

        logging.info("Using 2D UNet")
        model = UNetBuilder.unet3D_3SliceInput(self.pretrained_weights, self.loss_function, Dropout_On = True)

        #logging.info(model.summary())
        return model

    def train_model(self, model,TrainingDataGenerator,ValidationDataGenerator , steps_per_epoch, val_image_files ):
        '''
        Set Model callbacks such as: 
        - Early stopping (after the validation loss has not improved for 25 epochs
        - Checkpoints: Save model after each epoch if the validation loss has improved 
        - Tensorboard: Monitor training live with tensorboard. Start tensorboard in terminal with:
        tensorboard --logdir=/path_to/logs
        Args:
            model: The initialized U-Net model
            TrainingDataGenerator: The train data generator
            ValidationDataGenerator: The validation data generator
            steps_per_epoch: The number of train steps in one epoch
            val_image_files: List of validation files
        returns:
            The trained model and the checkpoint file path
        '''

        early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='auto', verbose=0)
        datasetname = self.path.rsplit("/",1)[1]
        checkpoint_filepath = (self.path + "/logs" + "/pretrained_weights" + datasetname + ".hdf5") 
        os.makedirs((self.path + "/logs"), exist_ok=True)
        model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor=('val_loss'), verbose=1, save_best_only=True)

        tensorboard = TensorBoard(log_dir = self.path + "logs/" + "/" + format(time.time())) 
        logging.info("Tensorboard log is created at: logs/  it can be opend using tensorboard "
                     "--logdir=logs for a terminal in the Project folder")
        callbacks_list = [model_checkpoint, tensorboard, early_stopping]

        '''
        Train the model given the initialized model and the data from the data generator
        '''
        model.fit_generator(TrainingDataGenerator,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=ValidationDataGenerator,
                                validation_steps=len(val_image_files),
                                max_queue_size=50,
                                epochs=self.epochs,
                                callbacks = callbacks_list,
                                use_multiprocessing=True)
        logging.info('finished Model.fit_generator')
        return model, checkpoint_filepath

    def test_set_evaluation(self, model, Training_Input_shapes, num_channels, Input_image_shape):
        '''
        Evalute the model on the testset
        Args:
            model: the trained or initialized model
            Training_Input_shape: The dimensions of the input data
            num_channels: Number of channels (e.g.: 3 for RGB)
            Input_image_shape: The shape of the input images
        returns: the results of the tested images, a list of filenames of the testset, the number of images tested
        '''
        test_image_files = os.listdir(os.path.join(self.path + "/test/image"))
        num_test_img = int(len(os.listdir(self.path + "/test/image"))) # TODO: check if needed

        '''
        Initialize the testset generator
        '''

        for i, test_image_file in enumerate(test_image_files):
            test_image_file = [test_image_file]
            Training_Input_shape = Training_Input_shapes[i]
            testGene = testGenerator(Training_Input_shape, self.path, num_channels, test_image_file, self.resize_factor)
            results = model.predict_generator(testGene, steps=1*(Training_Input_shape[0]),
                                              use_multiprocessing=False, verbose=1)

            '''
            Save the models prediction on the testset by printing the predictions 
            as images to the results folder in the project path
            '''
            saveResult(self.path, test_image_file, results, Input_image_shape, self.resize_factor)
        if self.evaluation == True:
            segmentation_regression_evaluation(self.path)

    def run(self):
        Input_image_shape = tuple((3, None, None, 1))  
        data_prepration_results = self.data_prepration()
        #nuclei_size = self.nuclei_size # TODO: check if needed
        Training_Input_shapes = data_prepration_results[0]
        num_channels = data_prepration_results[1]
        data_path = data_prepration_results[2]
        train_image_files = data_prepration_results[3]
        val_image_files = data_prepration_results[4]
        steps_per_epoch = data_prepration_results[5]
        if self.iterations_over_dataset != 0:
            TrainingDataGenerator, ValidationDataGenerator, _ = self.data_generator(data_path,
                                                                                Training_Input_shapes,
                                                                                num_channels,
                                                                                train_image_files,
                                                                                val_image_files)

        model = self.load_model()
        if self.iterations_over_dataset != 0:
            model, _ = self.train_model(    model,
                                            TrainingDataGenerator,
                                            ValidationDataGenerator ,
                                            steps_per_epoch,
                                            val_image_files  )

        self.test_set_evaluation( model,
                                        Training_Input_shapes,
                                        num_channels,
                                        Input_image_shape)

        model = None
    