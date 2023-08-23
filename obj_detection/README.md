# Object Detectaion

Code to identify rock fragments in ARW data. The code needs the file `data/Selected_wells.csv` containing information of all the wells with a  % Lithology greater of equal than 90%. 
The script `obj_dect_2.py` will use the `Selected_wells.csv` file to select images. 

# Description
The script can be run on a set of ARW images stored in a folder with the following structure
<pre>
Image
├── carbonates
├── mudstone
├── sandstone
└── volcanic</pre>
The path to the root folder, `Image`, can be passed as a input using the `--input_path` argument. The single rock images will be saved into the output folder keeping the same structure as the input folder.
Users can decide to run the code in parallel, using multiple processors. 

The script is designed to run one rock folder at the time.
Using input argument, user can controll the behaviour of the code. The full list of parameters is given below
<pre>
--input_path [./image/]
                        Path to the image to process. No default
  --rocks [ROCKS]       Choose rock type. Available options (case insensitive): carbonates, mudstones, sandstone, volcanic
  --np [1]              Number of process per image. Default value 1
  --rf [1.0]            Resize factor. Default value 0.2
  --output_path Default ./Output/Rocks
                        Path to Output Folder. If it does not exist, it will be created.</pre> 


# USAGE 

<pre> python obj_dect_2.py [-h] [--input_path [./image/]] [--rocks [ROCKS]] [--np [1]] [--rf [1.0]] [--output_path Default ./Output/Rocks] </pre>
or run <pre> python .\obj_dect_2.py --help  </pre> for more info.

# On HPC
Although it is possible, it is not advised to run this code on a local computer. You can use the provided batch script to run on HPC (See SBATCHs)

# Example
<pre> python obj_dect_2.py </pre>
Using the default setting, it will process file in  testimage/carbonates

#Requirements

See requirements.txt. To install <pre> pip install -r  requirements.txt </pre>




