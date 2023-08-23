"""
NAME: testing_model.py
    
DESCRIPTION:
Script to evaluate models on test dataset. It can be used to run only on a given folder for example to test carbonates only, by specifying the rocks type (--rocks Carbonates) or to the entire datasets (Default).

USAGE:
    python testing_model.py --rocks ALL --directory . --model_to_load <BEST_MODEL.hdf5>
    To run on HPC, please see Test_flow.sh

HISTORY:
    Created by: Magda Guglielmo <magda.guglielmo@csiro.au>
"""

#System import
import os
import argparse
#Tensorflow and Keras import
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
#local import
import load_model as lm

def parser():
    
    parser_ = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                    description='Analyse models',
                                    prefix_chars='--')
    parser_.add_argument('--rocks',
                       metavar='rocks',
                       type=str,
                       help='Rock type (initial). Options: C,M,S,V.')
    
    parser_.add_argument('--directory',
                       metavar='/path/to/model',
                       type=str,
                       help='path to model ')
    parser_.add_argument('--model_to_load',
                       metavar='model_filename.hdf5',
                       nargs='?',
                       const=1,
                       type=str,
                       default = None)
                               
    return parser_

def select_label(rocks):
    switcher = {
        'C':0,
        'M':1,
        'S':2,
        'V' :3,
        'ALL':-1
    }
    return switcher.get(rocks,"Not a valid Option")

def model_name(rocks):
    switcher = {
        'C':'CarbonatesVsAll',
        'M':'MudstoneVsAll',
        'S':'SandstoneVsAll',
        'V' :'VolcanicVsAll',
        'ALL':'Multiclass'
    }
    return switcher.get(rocks,"Not a valid Option")



def model_load(filename,binary=True,summary=False):
    '''
    Load saved tf model
    Input:
        filename: (str) path to the saved model
        summary: (bool) If yes print model summary
    Ouput:
        model: loaded model
    '''
    model = lm.load_model_from_hdf5(filename,compile=False)
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-5)
    if binary == True:
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        loss=tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=opt,
            loss=loss,
            metrics=['accuracy'])

    if summary:
        print(model.summary())
    return model

def get_metric(model,dts):
    '''
     Given a dataset (dts) evaluate loss and accurancy
     Input:
        model: tf model to evaluate
        dts: (tuple) in the form (image, label) dataset to evaluate
     Output:
        loss
        accuarancy

    '''
    if isinstance(dts,tuple):
        loss, acc = model.evaluate(dts[0], dts[1], verbose=2)
    else:
        loss,acc = model.evaluate(dts,verbose=2)
    return loss, acc
def create_dataset(image_list,N,N_class,label_to_exclude,shuffle =False):
    '''
        Given image_list containing all the image of interest, creates a x_train,y_train for
        binary classification (one vs all), with label_to_exclude defining object belong to class (0, one)
        Input: 
            image_list: (numpy array)  all the available images
            N: (int)lenght of image_list
            N_class: (int) number of classes
            label_to_exclude: (int) label of the classs choosen as the One.
            Shuffle:(bool) If true, the images,labels are shuffle
        Output:
            dataset : (numpy_array) images used for training (shuffled is shuffle ==True). It contains N//N_class for class 0 (one) and N//N_class object of the remaining class
            labels:  (numpy_array) 0,1 with 0 being the class of interest, 1 for the others.
                 
    '''

    #create a list of index for radom sampling
    #define first a list of label one for each of the remaining class. 
    #note that the indx should be the same as image_train
    labels = np.array([0]*(N//N_class) + [1]*(N//N_class) + [2]*(N//N_class) + [3]*(N//N_class))
    idx = np.arange(len(image_list))
    #create the one
    #IMPORTANT!!!!! THIS need to be tested
    one = image_list[(N//N_class)+(N//N_class)*(label_to_exclude-1):(N//N_class)*(label_to_exclude)+(N//N_class)]
     
    #now select all the label that don't match with the choosen one. In this case 3
    others_label = idx[labels != label_to_exclude]
    #sampling the index
    sample_index=np.random.choice( others_label,len(one), replace = False, p = None)
    #check
    if labels[sample_index].max() == label_to_exclude :
        print("Error with sampling")
        return None,None
    else:
        others_images = image_list[sample_index]
        dataset = np.concatenate((one,others_images),axis=0)
        labels = np.array([1]*len(one) + [0]*len(others_images))
        #shuffle
        if shuffle:
            idx = np.arange(len(dataset))
            np.random.shuffle(idx)
            dataset=dataset[idx]
            labels=labels[idx]
        return dataset, labels 
def predict_classes(prob, threshold = 0.5):
    '''
    Predict Classes for Binary classification. The classes 0 and 1 are assigned using threshold. E.g. if threshold ==0.5 
    all values with probability >= will be class 1, 0 otherwise
    Input:
        prob: numpy array. Probability as result of model.predict
        threshold: float. Default 0.5.
    Output
        pred: numpy array
    '''
    pred = prob.copy()
    pred[prob >=threshold]=1
    pred[prob<threshold]=0
    return pred

def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857 
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : 6]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        support.append(int(t[-1]))
        class_names.append(t[0])
        print('support: {0}'.format(support))

        try:
            check = float(t[1])
        except:
            t.remove(t[1])
            v = [float(x) for x in t[1:-1]]
            print(v)
            plotMat.append(v)
            print('Fix plotMat: {0}'.format(plotMat))
        else:
            v = [float(x) for x in t[1:-1]]
            print(v)
            plotMat.append(v)
            print('plotMat: {0}'.format(plotMat))



    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    #heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)
    return plotMat,xticklabels,yticklabels


def plot_summary(true,pred):
    
    s=classification_report(true, pred, target_names=['Carbonates','Mudstones','Sandstones' ,'Volcanics'])
    #s=s.replace("\n",' ')

    p,x,y=plot_classification_report(s)
    #plt.savefig('all_plot_classif_report.png', dpi=200, format='png', bbox_inches='tight')
    #plt.close()
    return p,x,y
def class_report(true,pred,class_):
    print(classification_report(true, pred, target_names=class_))


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None,cmap=plt.cm.Blues,filename=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    return ax

if __name__ == "__main__":
    #model info
    parser_=parser()
    args = parser_.parse_args()
    directory = args.directory
    label_ = select_label(args.rocks)
    #load model 
    model_dir=os.path.normpath(directory)
    if args.model_to_load is not None:
        filename = os.path.join(directory,args.model_to_load)
    else:
        #load the last/best model
        model_files = [ f for f in  os.listdir(model_dir) if f.endswith('hdf5')]
        filename = os.path.join(directory,model_files[-1])

    print('\n\nWorking with model {}\n\n'.format(filename))
    model = model_load(filename,False)
    
     
    #Read testing/validation test

    N_image_training = 40000
    N_image_val = 4000
#    N_image_test =4240 #small
    N_image_test = 21652
    N_class = 4 #number of class in the original dataset

    #data_dir=os.path.normpath('../../NpyArrays')
   
    base_path = './Images_min55px'
    testing_data_dir = os.path.join(base_path,'test_dataset')
    batch_size = 128
    Isize = 256
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        testing_data_dir,
        target_size=(Isize,Isize),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical',
        #subset = 'validation',
        )




    print("working with {}".format(label_))
    #Model metric 
    loss,acc=get_metric(model,test_generator)
    print('\n\n\nOverall metric on Validation:\n Acc {}\n Loss {}\n'.format(acc,loss))


    y_pred_prob = model.predict(test_generator)
     
    y_pred_sparse = np.argmax(y_pred_prob,axis=1)
    y_true_sparse = test_generator.classes
    class_report(y_true_sparse,y_pred_sparse,class_=['Carbonates','Mudstones','Sandstones','Volcanics'])
#    p,x,y=plot_summary(y_true_sparse,y_pred_sparse)
    plot_confusion_matrix(y_true_sparse, y_pred_sparse,['Carbonates','Mudstones','Sandstones','Volcanics'] ,
                          normalize=True,
                          title='AllVsAll',
                          cmap=plt.cm.Blues,filename='CM_AllvsAll.png')
    #Null test (?)
#    carbonates = image_testing[0:3605]
#    mudstone   = image_testing[3605:7210]
#    sandstone  = image_testing[7210:10815]
#    volcanic   = image_testing[10815:14420]
#    
#    cc = model.predict(carbonates)
#    #cc=np.argmax(cc,axis=1)
#    cc=np.max(cc,axis=1)
#    mm = model.predict(mudstone)
#    mm=np.max(mm,axis=1)
#
#    ss = model.predict(sandstone)
#    ss=np.max(ss,axis=1)
#    vv = model.predict(volcanic)
#    vv=np.max(vv,axis=1)
#
#    model_one = model_name(args.rocks)
#
#    plt.figure()
#    plt.subplot(2,2,1)
#    plt.hist(cc,bins=100)
#    plt.title('_'.join([model_one,'Carbonates as input']))
#
#
#    plt.subplot(2,2,2)
#    plt.hist(mm,bins=100)
#    plt.title('_'.join([model_one,'Mudstone as input']))
#
#    plt.subplot(2,2,3)
#    plt.hist(ss,bins=100)
#    plt.title('_'.join([model_one,'Sandstone as input']))
#
#    plt.subplot(2,2,4)
#    plt.hist(vv,bins=100)
#    plt.title('_'.join([model_one,'Volcanic as input']))
#    plt.title('_'.join([model_one,'Volcanic as input']))
#
#    plt.tight_layout()
#    plt.savefig(model_one+'.png')
