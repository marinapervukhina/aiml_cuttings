# MODEL TRAINING AND EVALUATION

This folder contains all the necessary script to create and train the network, as well to test the model on an unknown dataset. The model has been used to train images with a minimum size of 55 px per side. 


## Folder structure
<pre>
</pre>
<pre>
├── README.md
├── SBATCHs
├── example_plots
├── load_model.py
├── model_training_flow.py
├── plot_loss.py
├── requirements.txt
└── testing_model.py
</pre>
### SBATCH example to run on HPC
<pre>
├── SBATCHs
│   ├── Model_flow.sh
│   └── Test_flow.sh
</pre>
### Image examples
<pre>      
├── example_plots
│   ├── CM_AllvsAll.png
│   ├── loss.png
│   └── training_loss_acc.png
</pre>
# Requirements
See requirements.txt
Note that the requirements are already satisfied on HPC. See module load in SBATCH

# AUTHORS

Magda Guglielmo 
Brint Gardner
Arun Sagotra