   
##
# @file obj_detc.py
# @brief Rock chips detection 
# Given an image with "default background", the code identifies rocks chips and saves single rocks images
# 
# Modified version of obj_det.py. The current version is designed to run on Bracewell and it uses  the multiprocessing Pool to analyse image in parallel.
#
# @section obj_det_imports Requirements
# @subsection obj_det_imports_1 Image processing
# - CV2
# - skimage feature 
# - imageio (used to read image not supported by cv2)
# - rawpy  (able to read image in raw format)
# @subsection obj_det_imports_2 Analysis
# - numpy
# - pandas
# - scipy.signal (analysis of the image peaks)
# - matplotlib for basic plots (used only for debug purpose)
# @subsection obj_det_imports_3 local library
# - impath local library to manage image path
# @section obj_det_bkg Background masking
# All the images used in this project have a blue-green background (default). Initially, the background removal used a tuple of HSV values corresponding to the background colour. However, 
# reflected light transfers the background colour on the rocks, and this approach left black pixels on the images of the single chip. 
# To address this problem, a new method is proposed that identifies and masks the background using the histogram of the grey image. The minimum point between the histogram peaks gives the RGB value to mask.
# Note: the proposed colour space (see function description) only works for images with the default background. An appropriate colour space needs to be defined if the background will change in future. 
#
# @section obj_det Rocks Detection 
#  Edge detection is performed on the original image after masking the background using the Canny algorithm (note previously, we used a Sobel filter). To ensure a smooth background, the image is equalized using  Contrast Limited Adaptive Histogram Equalization (CLAHE).
#
# @section obj_edge Contour Scaling and Cleaning.
# The proposed approach allows to work on a smaller image to identify the object and then to use the contours to "cut" the rocks from the original (higher resolution) images.
# Therefore, the contours found on the smaller image need to be scaled back to the actual image size. 
# In addition, when a box is created around the chips is possible that nearby fragments (outside the contour of interest) are selected as well. The function clean_outside_contour ensures that all objects outside the contour of interest are removed. 
# Note: multiple rocks in a single box might not be a problem if they are all of the same class. Confusion arises when multiple rocks type might be present in the same image.
#
# @section obj_det_processing Image processing
# The image_processing function acts as an orchestrator. It opens the image file (using a different method according to the file extension) and reads its content making sure that the final array uses the cv2 default colour space.
# If the user decides to work on a rescaled image then this function proceeds to rescale the image, remove the background and identify the object by calling the corresponding functions. An extra check is added to remove "dusted" images (images of chips too small to be detected with the given resolution).
# The final single rock chips are saved. Note: there is a check on the contour area that guarantees that objects with an area greater than a threshold (chosen to be the area of a circle of radius 10 pixels).
# @section callback_author Author(s)
#
# - Magda Guglielmo (IMT, Eveleigh) <magda.guglielmo@csiro.au>
# - Gardner, Brint (IM&T, Clayton) <Brint.Gardner@csiro.au>
# - Date: 24/03/2022
# - Last updated by: Magda Guglielmo
# - Last update date: 06/07/2022
#
##


# Imports

import os
import time
import argparse
from multiprocessing import Pool,current_process

import numpy as np
import pandas as pd
import cv2
import imageio as iio
import rawpy
from skimage import feature
from scipy.signal import find_peaks,savgol_filter
from utils.impath import check_path_output,get_path_image


def parser():
    parser_ = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                    

                                    description='Object Selection',
                                    prefix_chars='--')
  

    parser_.add_argument('--input_path',
                   metavar='./image/',
                   nargs='?',
                   const=1,
                   type=str,
                   default = './testimage',
                   help='Path to the image to process')

    parser_.add_argument('--rocks',
                       type=str,
                       nargs='?',
                       const=1,
                       default='carbonates',
                       help='Choose rock type. Available options (case insensitive): carbonates, mudstones, sandstone, volcanic')
    parser_.add_argument('--np',
                       metavar='1',
                       type=int,
                       nargs='?',
                       const=1,
                       default=1,
                       help='Number of process per image. Default value 1')
    parser_.add_argument('--rf',
                       metavar='1.0',
                       type=float,
                       nargs='?',
                       const=1,
                       default=1.0,
                       help='Resize factor. Default value 0.2')
    parser_.add_argument('--output_path',
                       metavar='Default ./Output/Rocks',
                       type=str,
                       default = './Output/Rocks',
                       help='Path to Output Folder. If it does not exist, it will be created. ')
                               
    return parser_

#Background masking
def clean_background(image,show=False):
    '''
        This function aims to identify the background and create a maked image with only rock. The assumption are:
            1. The bacground of the pictures has a prevalence of blue colour (true for the current dataset), while the rocks have low blue. 
            2. The bacground has low red, while rocks are high in red.
        For the above reason, the background (and rocks) can be identified in a red / blue space. The histogram of the image in this space will have
        at least 2 peaks idicating background and rocks. But selecting all pixels with values higher than the colour corresponding to the minimum between the first two peaks, the bacground is remove.
        Input : 
            image : BGR image (default for cv2)
        Output:
            image_no_background : BGR image no background of type uint8. 
    '''
    b,g,r = cv2.split(image)  
    #make sure b is not 0 to avoid divide for 0 errors
    b = ((b/65535)*255).astype('uint8')
    r = ((r/65535)*255).astype('uint8')
    
    b[b==0] =  1
    r[r==0] =  1
   
    
    image_rb_space =r/b
    

    for n in [100,500,1000,10000]:
        hist, bin_edges = np.histogram(image_rb_space,bins=n)# find histogram
        x = np.arange(len(hist))
        y=hist
        y=y/y.max()
        y = savgol_filter(y,21,3)
        y[y<0.005] = 0
        peaks, _ = find_peaks(y, height = 0.005,distance=20) #index does not correspond to x anymore
        i = np.argsort(y[peaks])[::-1]
        peaks = peaks[i]

        try:
            idx_tp  = np.argmin(y[peaks[0]:peaks[1]])+peaks[0]#different sorting
            turning_point  = bin_edges[idx_tp+1]
            break
        except Exception as e:
            if n==10000:
                return None
            else:    
                continue

    image_no_background = np.zeros(image_rb_space.shape)
    image_no_background[image_rb_space  > turning_point]=1
    return image_no_background


#Rocks Detection
def find_contour(image):
    kernel_f = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.erode(image,kernel_f,iterations = 1)
    contours,_= cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return contours
#scale_contour
def scale_contour(cnt,scale_factor,padding=None):
    '''
    Recenter/Translate contour from low to high resolution
    Input:
        cnt: numpy array. Contour in lower resolution
        scale_factor: double. Resize factor used to resize original image
        padding : tuple defualt 
    Output:
        cnt_nomr: numpy array. Object contour in high resolution
    '''
    if padding is None:
        cnt_norm = (cnt*scale_factor).astype(np.int32)
        xmin = min(cnt_norm[:,0,0])
        ymin = min(cnt_norm[:,0,1])
        cnt_norm[:,0,0] = cnt_norm[:,0,0]-xmin 
        cnt_norm[:,0,1] = cnt_norm[:,0,1]-ymin
    else:
        cnt_norm =cnt
        xmin = min(cnt_norm[:,0,0])-padding[0]
        ymin = min(cnt_norm[:,0,1])-padding[1]
        cnt_norm[:,0,0] = cnt_norm[:,0,0]-xmin 
        cnt_norm[:,0,1] = cnt_norm[:,0,1]-ymin
        
    return cnt_norm.astype(np.int32)


def clean_outsize_contour(image, contour):
    '''
        Remove objs from frame outside contour of interest
        Input:
            image: image object (frame) to process
            contour: edge of interest
        Output:
            image: cleaned image
    '''
    img = image.copy()
    mask =  np.zeros(img.shape[:-1]).astype(np.uint16)
    cv2.fillPoly(mask, [contour], 65535)
    selected = mask != 65535
    img[selected] = 0
    return img,mask
def quality_check(image):
    '''
        This function quantifies the quality of the single rock images by checking the percentage of background pixels (e.g. blue/green shades) that have no be removed properly. 
        Ideally, after the background removal, this percentage should be small, and the quality high. It is used to avoid running the model on "bad images"
    '''
    im = ((image/65535)*255).astype('uint8')
    to_HSV=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    hue,_,_=cv2.split(to_HSV)
    hue[hue >80] = 0
    mask_black = np.zeros(hue.shape)
    mask_black[hue==0]=1
    quality = 1 - mask_black.sum()/np.prod(mask_black.shape)
    return quality
# Main Routine
def image_processing(image_filename,resize_perc, background = True, output_path=None):
    '''
     Reads and processes  image file. After saving the image, it returns a dataframe with some information about the created images.
     Input:
        -image_filename: str. Path to the image to process
        -resize_perc: float number. User defined resize factor. A value of 1 implies working on the original images (no resize.). Note, for ARW file the suggested value is 0.4 lower value reduce edge detection performance. 
        - background: bool. Default True. If false, no correction to the region outside the contour is performed.
        - output_path: string, default None. If None, and save is true the file will be saved in the working directory. 
          Note it assume that the path to the images is something like rocks_type/well/image

  
    '''    
    folder = image_filename.rsplit(os.sep,3)[-3:]
    rocks_folder = folder[0]
    well = folder[1]
    image_id = folder[2].rsplit('.')[0]
    file_basename = '_'.join([well,image_id,'obj_'])
    if output_path is None:
        output_path = os.path.join(os.getcwd(),"Output", rocks_folder)
        check_path_output(output_path)
    filename =  os.path.join(output_path,file_basename)
    
    #check how to read the image
    if image_filename.endswith('ARW'):
        with rawpy.imread(image_filename) as raw:
            rgb = raw.postprocess(no_auto_bright=True, output_bps=16)
        im =cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        im = cv2.imread(image_filename,cv2.IMREAD_UNCHANGED)
        if type(im) is not np.ndarray:
            im = iio.imread(image_filename)
            im=cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    
    #check if you want to resize the image
    if resize_perc <1:
        #Resize by resize_perc
        width  = im.shape[1]
        height = im.shape[0]
        RW = int(width*resize_perc)
        RH = int(height*resize_perc)
        im_work = cv2.resize(im,(RW,RH))
    else:
        im_work = im.copy()
    
    #remove background
    
    im_work = clean_background(im_work,image_filename) #16 bits
    if im_work is None:
        print('File {} (class: {}) histogram problem. im_work is NULL'.format(image_filename,rocks_folder))
        return []
    
    
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    im = cv2.filter2D(src=im, ddepth=-1, kernel=kernel)
  
    #check if "dust"

    white =  (im_work >0).sum()
    black =  (im_work ==0).sum()
    dfs=[]
    if white / black < 0.7:
        
        contours=find_contour(im_work)

        sorted_contours = sorted(contours, key=cv2.contourArea, reverse= True)

        min_area = 200;#area that are bigger than a circle of radius 10 px. This is a loose condition used to avoid saving objs with very small area
        ca = [cv2.contourArea(c) for c in sorted_contours if cv2.contourArea(c) ]
        sorted_contours =[c for c in sorted_contours if cv2.contourArea(c) >min_area]
        
        df = pd.DataFrame(columns = ['RockType','well','imageId','w','h','px_in_c','c_area_Green','path'])
        row = 0
        
        for (i,c) in enumerate(sorted_contours):
            x,y,w,h= cv2.boundingRect(c*int(1/resize_perc))
            cc = scale_contour(c, int(1/resize_perc))
            #Add padding to box
            padding_or = int(0.08*max([h,w]))
            if y < padding_or or y+h+padding_or > im.shape[0]:
                padding_y = 0
            else:
                padding_y = padding_or
            if x < padding_or  or x+w+padding_or > im.shape[1]:
                padding_x = 0
            else:
                padding_x = padding_or
            obj = im[y-padding_y:y+padding_y+h, x-padding_x:x+w+padding_x]
            
            if background:
                    cc2 = scale_contour(cc, int(1/resize_perc),(padding_x,padding_y))
                    obj,_= clean_outsize_contour(obj,cc2 )
                    gray = cv2.cvtColor(obj,cv2.COLOR_BGR2GRAY)
                    black = (gray ==0).sum()
                    #print(i,(black/np.prod(gray.shape)))
                    quality = quality_check(obj)
                    if (black/np.prod(gray.shape)) < 0.8:
                        quality = quality_check(obj)
                        if quality >0.3:
                            obj_id = filename + str(i).zfill(4) +".png"
                            cv2.imwrite(obj_id,obj)
                            df.loc[row,'RockType']=rocks_folder
                            df.loc[row,'well']=well
                            df.loc[row,'imageId']=image_id
                            df.loc[row,'px_in_c']=np.count_nonzero(gray)
                            df.loc[row,'c_area_Green'] = cv2.contourArea(cc)
                            df.loc[row,'w']=w
                            df.loc[row,'h']=h
                            df.loc[row,'path']=obj_id
                            row=row+1

            else:
                    gray = cv2.cvtColor(obj,cv2.COLOR_BGR2GRAY)
                    black = (gray ==0).sum()
                    if (black/np.prod(gray.shape)) < 0.8:
                        cv2.imwrite(obj_id,obj)

        if not df.empty:
            #path_df = os.path.join(output_path,'tmp',image_id+'.csv')
            #df.to_csv(path_df,index=False)
            dfs.append(df)
    else:
        print('File {} dust or obj too small to be detected'.format(image_filename))
    return dfs

if __name__ == "__main__":

    
     __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
     parser_ = parser()
     args = parser_.parse_args()
     
     if args.input_path is None:
        
        print("Input Path missing. Usage:")
        print('''\t\t--input_path [./image/] Path to the image to process\n
                --rocks [ROCKS]  Choose rock type. Available options (case insensitive): carbonates, mudstones, sandstone, volcanic\n
                --np [1]              Number of process per image. Default value 1 \n
                --rf [1.0]            Resize factor. Default value 0.2 \n
                --output_path Default ./Output/Rocks\ Path to Output Folder. If it does not exist, it will be created.\n''')
        import sys
        sys.exit()
     print('Starting creating image_path')
     output_path = args.output_path
     if not os.path.exists(output_path):
         os.mkdir(output_path)
     input_path = args.input_path
     
        
     if args.rocks:
         input_path = os.path.join(input_path,args.rocks)
         output_path = os.path.join(output_path,args.rocks)
     image_path=get_path_image(input_path,skip=True)
     print('Total of  {} images to process'.format(len(image_path)))
      #Select well where purity is >=90: the file Selected_wells.csv has been created offline
     wells = pd.read_csv('data/Selected_wells.csv')
     #create dataframe from image_path
     df_images=pd.Series(image_path).str.rsplit(os.sep,3,expand=True)
     df_images.columns=['path','RockType','ImageFolder','filename']
     df_images['path']=image_path
     df_images=df_images.merge(wells,on='ImageFolder',suffixes=(None,'_'),indicator=True)
     #reset image_path so that only the "correct" image will be analyses.
     image_path = df_images['path'].tolist()

    
     print('Done with image_path {}'.format(output_path))
     check_path_output(output_path)
    
     Np =args.np 
     if Np > 1: 
         print("Start Pool") 
         dfs = []
         Numprocess = Np 
         st=time.perf_counter() 
         P=Pool(Numprocess) 
         for i in range(len(image_path)): 
             p=P.apply_async(image_processing, args=(image_path[i],args.rf,True,output_path)).get() 
             dfs.extend(p)
         P.close() 
         P.join() 
         et=time.perf_counter() 
         print("To Process {} files tooks {} minutes (average time per file {} minutes)".format(len(image_path),(et-st)/60,(et-st)/len(image_path)/60))
     
     else: 
         st=time.perf_counter() 
         dfs=[]
         for i in range(0,64):#len(image_path)): 
             dfs.append(image_processing(image_path[i],args.rf,True,output_path) )
         et=time.perf_counter() 
         print("To Process {} files tooks {} minutes (average time per file {} minutes)".format(len(image_path),(et-st)/60,(et-st)/len(image_path)/60))
     
     df = pd.concat(dfs)
     df.to_csv(os.path.join(output_path,args.rocks+'_info.csv'),index=False)
     print("Done")
