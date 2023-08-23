# AI/ML based cuttings analysis

Code developed during scientific collaboration projects ERRFP-1777 (Demo1) and ERRFP-1202 (Demo2).
The projects aimed to developed a tool to classify rock chips from a set of labelled images. 
The classification is perfomed using CNN developed during the collaboration. The code in this repository can be used to create the image datasets for training and testing, create the CNN used for classification as well as for testing new set of image (see Demo1 - Demo2 folders)

# Repo structure
<pre>
├── Demo1
├── Demo2
├── README.md
├── Training
├── docs
└── obj_detection
</pre>

## The obj_detection folder 
</pre>
obj_detection
    ├── Output
    ├── README.md
    ├── SBATCHs
    ├── data
    ├── obj_dect_2.py
    ├── requirements.txt
    ├── testimage
    └── utils
</pre>

Example code to create training and test datasets. The code can be run on HPC using the SBATCH file provided in the `SBATCHs` folder. Use the `obj_detection/README.md` for more 
## Training
<pre>
Training
│   ├── README.md
│   ├── SBATCHs
│   ├── example_plots
│   ├── load_model.py
│   ├── model_training_flow.py
│   ├── model_training_medium_large.py
│   ├── plot_loss.py
│   ├── requirements.txt
│   └── testing_model.py
</pre>
Code to train and test the example. Use the `training/README.md` for more info

## The Demo folders

Two demos have been developed that can be used to present the method. The code structure is the same for both, but they differ in classification method. `Demo1` uses only 1 model to classify rock of size greater than 55 px. Small rocks will ignore from the analysis (Code developed ERRFP-1777). 
Conversely, `Demo2` extend the functionality by including small objects. The classification uses two different models depending on the size of the rock chip image. 
See corresponding README.md.

# Requirement
<pre> pip install -r requirements.txt </pre>

**_Note_** The code in each subfolder can be run idependently and a separate requirements.txt file is provided. The one in the root folder is general and it will install all the necessary package.

# Author
Magda Guglielmo
Brint Gardner
Arun Sagotra
