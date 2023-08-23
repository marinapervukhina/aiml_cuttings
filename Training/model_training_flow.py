"""
NAME : 
    model_training_flow.py

DESCRIPTION

    This script will create  the network and training it using training and validation datasets. It uses ImageDataGenerator flow_from directory to create the dataset, the root folder must have the images organised by class. For example:
    NOTE: The same network can be applied to the original dataset (Images_min55px, ERRFP-1777) and for the grain/small dataset (minimum size 25px max size 99px, but make sure to change the size of the input layer (see code comments)). 
    Make sure that the right path to images is used.
    Images_min55px:
    |__training_dataset
        |__carbonates
        |__mudstone
        |__sandstone
        |__volcanic
    |__validation_dataset
        |__carbonates
        |__mudstone
        |__sandstone
        |__volcanic 
    With this structure, the label of the images is automatically done by the datageneration fucntion.
    
    It also create the convolution neural network using Swish activation function. The main difference from the previous model is that it uses two dense layer at the end of the network.

    The model uses EARLYSTOP with 20 epochs buffer to avoid overfitting. It also save models that show improvement in the loss function (model checkpoints). All the models are saved in the same folder, together with the metric plot (loss.png)
USAGE
    To run on HPC please see, Model_flow.sh
    
HISTORY:
    Created by: Magda Guglielmo <magda.guglielmo@csiro.au>
    Created at: August 2022
    Modified at: 5 October 2022

"""


#System import 
import os
#tensorflow import
import tensorflow as tf
from tensorflow.keras import datasets, layers, models,regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Keras import
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
#Matplotlib to show results
import matplotlib.pyplot as plt

def swish(x):
    """Create Swish activation 

    Args:
        x (tensor): Input tensor

    Returns:
        activation function: Swish activation function applied to x
    """
    return tf.nn.swish(x)

def create_model(output=4,size=256):
    """Create the ConvNet

    Args:
        output (int, optional): Classes. Defaults to 4.
        size (int, optional): Size of the input image. Defaults to 256.

    Returns:
        model: Sequential model
    """
    momentum=0.95
    get_custom_objects().update({'swish':tf.keras.activations.swish})
    activation = 'swish'
    model = models.Sequential()
    model.add(layers.Input((size,size,3)))
    #model.add(layers.experimental.preprocessing.Rescaling(1/65535))
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation=activation, kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.BatchNormalization(momentum=momentum,trainable=True))
    model.add(layers.Conv2D(64, (3,3), padding="same", activation=activation))
    model.add(layers.Conv2D(64, (3,3), padding="same", activation=activation))
    #model.add(layers.Conv2D(128, (3,3), padding="same", activation='relu'))
    #model.add(layers.Conv2D(256, (3,3), padding="same", activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    #model.add(layers.BatchNormalization(momentum=momentum,trainable=True))
    model.add(layers.Conv2D(128, (3,3), padding="same", activation=activation))
    model.add(layers.Conv2D(128, (3,3), padding="same", activation=activation))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Conv2D(256, (3,3), padding="same", activation=activation))
    model.add(layers.Conv2D(256, (3,3), padding="same", activation=activation))#,kernel_regularizer=tf.keras.regularizers.L2()))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256,activation=activation))
    model.add(layers.Dropout(0.5))     
    model.add(layers.Dense(128,activation=activation))
    model.add(layers.Dense(output, activation='softmax'))
    print(model.summary())
    return model

def def_model(train,val,batch,epoch,Ntrain, Nval,output=4,size=256):
    """Define the models.
    This function  create the Sequential model using create_model(output,size) function. It defines the optimizers and the callbacks of the models andthe proceed to compile and fit the model

    Args:
        train (dataset): Dataset of image used for training. Create from folder using create_dataset() function 
        val (dataset): Dataset of image used for validation. Create from folder using create_dataset() function
        batch (int): Batch size.
        epoch (int): Number of epochs to train
        Ntrain (int): Size of the training datasets
        Nval (int): Size of the validation datasets
        output (int, optional): Number of classes in dataset. Defaults to 4.
        size (int, optional): Size of the input layer. Defaults to 256.

    Returns:
        models:Train models
        history: model history
    """
    ilr = 0.01 #initial learning rate
    ds = 5.0 #decay steps (1/t)
    dr = 0.5 # decay rate 
    lr_fn = tf.keras.optimizers.schedules.InverseTimeDecay(ilr,ds,dr)
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    #norm = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1, dtype=None, mean=None, variance=None)
    #norm.adapt(x_train) 
    #x_train = norm(x_train)
    model = create_model(output,size)
    #model.compile(optimizer=opt,
    #        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
     #       metrics=['accuracy'])
    model.compile(optimizer=opt,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])
 
    #Callback to adj lr 
    rlronp=tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=3,min_lr=0.00001,verbose=1,min_delta=0.1,cooldown=3)
     #callback to save best model
    modelfilename = 'train_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
    mdc = tf.keras.callbacks.ModelCheckpoint(filepath=modelfilename, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',patience=20,verbose=1)
    logger = tf.keras.callbacks.CSVLogger('model_history.csv',append=True)

    history=model.fit(train,
                      batch_size=batch,
                      epochs=epoch, 
                      steps_per_epoch=Ntrain//batch,
                      validation_data=val,
                      callbacks=[mdc,es,logger])
    
    return model, history

def run(train,val,batch,epoch,N_image_training, N_image_val,output=4,size=256):
    """Function to run the models
    Args:
        train (dataset): Dataset of image used for training. Create from folder using create_dataset() function 
        val (dataset): Dataset of image used for validation. Create from folder using create_dataset() function
        batch (int): Batch size.
        epoch (int): Number of epochs to train
        Ntrain (int): Size of the training datasets
        Nval (int): Size of the validation datasets
        output (int, optional): Number of classes in dataset. Defaults to 4.
        size (int, optional): Size of the input layer. Defaults to 256.

    Returns:
        models:Train models
        history: model history
    """
    model, history = def_model(train,val,batch,epoch,N_image_training, N_image_val,output,size)
    return model, history

def plot_metrics(history,a=0,b=1,save=False, filename=None):
    """Plot models metrics. Loss and accurancy

    Args:
        history (dict):Model history
        a (int, optional): Lower limit of the y-axis, used for loss plot. Defaults to 0.
        b (int, optional): Upper limit of the y-axis, used for loss plot. Defaults to 1.
        save (bool, optional): If true, will save the plots. Defaults to False.
        filename (str, optional): Filename to save the plots. Must be given is save is True. Defaults to None.
    """
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.ylim([a, b])
    #yticks=np.arange(a,b+0.2,0.2)
    #step = (max(yticks)+0.2 )/0.2
    #xstep =( len(history.history['accuracy'])-1 ) / step
    #xticks =np.arange(1, len(history.history['accuracy'] ) +xstep,xstep)
    plt.minorticks_on()
    #plt.yticks(yticks)
    #plt.xticks(xticks)
    
    plt.legend(loc='lower right')
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label = 'val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #yticks=np.arange(0,1.2,0.2)
    #step = (max(yticks)+0.2 )/0.2
    #xstep =( len(history.history['accuracy'])-1 ) / step
    #xticks =np.arange(1, len(history.history['accuracy'] ) +xstep,xstep)
    plt.minorticks_on()
    #plt.yticks(yticks)
    #plt.xticks(xticks)
    plt.ylim([0, 1])
    plt.tight_layout()
    if save and filename is not None:
        plt.savefig(filename)

def create_datagen():
    """ Create datagenerator from folder. Any pre-process to the images must be passed here.

    Returns:
        generator: for training and validation.
    """
    train_datagen =ImageDataGenerator(
        rescale=1. / 255,
    #add prepocess here  
    #validation_split=0.50
    )

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    val_datagen = ImageDataGenerator(
        rescale=1./255,
    #validation_split=0.6432,
    )
    return train_datagen, val_datagen

def create_dataset(train_data_dir, validation_data_dir,testing_data_dir, img_height, img_width, batch_size,ALL=False,scope='train'):
    """ Create dataset for folders. It uses the generator created by create_datagen()

    Args:
        train_data_dir (str): Path to the training images
        validation_data_dir (_type_):  Path to the validation images
        testing_data_dir (_type_): Path to the testing images
        img_height (int): Height of the images in folders
        img_width (int): Width of the images in folders
        batch_size (int): Batch size
        ALL (bool, optional): If True, returns training, validation and testing dataset, otherwise only training and validation datasets.. Defaults to False.
        scope (str, optional): Option train and test. Defaults to 'train'.

    Returns:
        _type_: _description_
    """
    train_datagen,val_datagen =create_datagen()
    if ALL:
        scope = 'train test'
    if 'train' in scope:     
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            shuffle=True,
            class_mode='categorical',
            #subset = 'training',
            )

        validation_generator = val_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            shuffle=True,
            class_mode='categorical',
            #subset = 'training',
            )
       
    if 'test' in scope:    
        test_generator = val_datagen.flow_from_directory(
            testing_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            shuffle=True,
            class_mode='categorical',
            #subset = 'validation',
            )
    if ALL:
        return train_generator, validation_generator,test_generator
    else:
        return train_generator, validation_generator





base_path = './Image_min55px' #root path of directory containing training, validation and testing dataset
train_path = os.path.join(base_path,'train_dataset')
val_path = os.path.join(base_path,'val_dataset')


batch_size =128
Isize = 256 # input size for the 55 px sample
#Isize = 64 # input size for the grain small (25px<=size<100px) sample
import time
s=time.perf_counter()
ds_train, ds_validation = create_dataset(train_path, val_path,None, Isize, Isize, batch_size)
e=time.perf_counter()
print('Time create_dataset {}'.format(e-s))



epoch = 200 #Maximum Number of epochs. The model uses earlystop to avoid overfitting. 
N_image_training =120000

N_image_val = 20000
model,history=run(ds_train,ds_validation,batch_size,epoch,N_image_training, N_image_val,output=4,size=Isize)
plot_metrics(history,a=0,b=max(history.history['val_loss']),save=True, filename='loss.png')
#loss,acc=tm.get_metric(model,ds_test)

