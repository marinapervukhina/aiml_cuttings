##
# @file image_path.py
# @brief Routine to create, clean and manage the image path.
# @section imp_get_path Get Path of images
# Given a path to the folder containing the data returns all images file in the folder and its subfolder
#@section imp_check_output Check output directory
# When the output directory is passed, the function check whether the folder exists. If not will create it using the specified path
#@section imp_cleaning Remove unwanted images
# Given a list of path, the function checks if the files have a different name the others. For example, when the object detections labelled a file as dust (object too small for the image resolutions) the user my decided to 
# remove them from the training test. The output is a list of image paths that have format IMAGEID_OBJID.png, for example _DSC0004_obj_0000.png, where IMAGEID = _DSC004 OBJECTID=obj_0000 
#
import os

def get_path_image(path_file=None,folder="data",skip=False):
    '''
        Get file in a folder
    Input: 
        folder: str. Name of folder within current path
    Ouput:
        path_to_image: list of all file in folder and its subfolder
    '''
    if path_file is None:
        path_file=os.path.join(os.getcwd(), folder) #\\Carbonates\\Poseidon1_2310-2320\\", "_DSC0020_PSMS16.tif"
    
    path_to_image=[]
    if skip:
        for root, dirs, files in os.walk(path_file):
            for file in files:
                    path_to_image.append(os.path.join(root,file))
        #check if there are ARQ. If so, take the corresponding ARW file
        ARQtoARW = [file.rsplit('_',1)[0]+'.ARW' for file in path_to_image if file.endswith('ARQ')]
        if len(ARQtoARW) >0:
            p=[path_to_image[path_to_image.index(a)] for a in ARQtoARW]
        #if no ARQ files are found. Select all the ARW file and sampling with a frequency of 16
        else:

            p=[pp for pp in path_to_image if pp.endswith('ARW')]
            p=p[::16]
                

            
        path_to_image = p
    else:
        for root, dirs, files in os.walk(path_file):
            for file in files:
                if file[-3:].lower() in ['tif','arw','png','jpg']:
                    path_to_image.append(os.path.join(root,file))
    return path_to_image

def check_path_output(folder):
    '''
        check if a given folder exsists in path. If not if creates
        Input: 
            folder: str. Path to folder
        Output:
            None
    '''
    path_folder=folder
    isExist = os.path.exists(path_folder)
    if not isExist:
        try:
            os.makedirs(path_folder)
        except Exception as e:
            print('{} causes error {}'.format(path_folder,e))

    pass

def cleaning(list_to_clean):
    '''
        remove image used for checking or labelled as dust
        Input:
            list_to_clean: list of image path
        Output:
            list with only path to obj image
    '''
    noobj=[b for b in list_to_clean if b[-5] == '_' or 'dust' in b]
    for n in noobj:
        list_to_clean.remove(n)
    return list_to_clean
