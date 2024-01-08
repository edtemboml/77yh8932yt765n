###############################################################################################################################
### Created By : Edwin Tembo
### Purpose    : 
### Simple Script for Image Quantization. Reduces the number of colors in an image by picking the most relevant colors through 
### K-Means Clustering. This can reduce the storage size of images when the dimensions of the image need to be kept constant.
################################################################################################################################
import cv2
import argparse
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
import os

DEFAULT_ALGO               = 'lloyd'
DEFAULT_RANDOM_STATE       = 32
DEFAULT_MAX_OUTPUT_SIZE_MB = 5.8
HOME_PATH = os.path.expanduser('~')
DEFAULT_FILES_PATH = os.path.join(HOME_PATH,'FILEPATH')

def cust_split(str):
    return [char for char in str]


def imageResizeOnly(image_path):
    resized_image =resize_image(image_path)
    image_name = image_path.split('/')[-1]
    new_image_name = f'resized_{image_name}'
    
    path = Path(image_path)
    parent_dir = path.parent.absolute()
    print(os.path.join(parent_dir, new_image_name))
    cv2.imwrite(os.path.join(parent_dir, new_image_name), resized_image) 


def resize_image(image_path , max_size_mb=5.8, scaling_factor = None ,   interpolation=cv2.INTER_AREA):
    
    if scaling_factor == None:
        file_size_mb = os.stat(image_path).st_size  / (1000*1000)
        print(f'file size mb{file_size_mb}')
        if file_size_mb > max_size_mb:
            scaling_factor = max_size_mb/(file_size_mb)
            print(f'scaling_factor : {scaling_factor}' )
            img=cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    print('Original h,w : ',img.shape)
    width       = int(img.shape[1] * scaling_factor)
    height      = int(img.shape[0] * scaling_factor)
    scaled_dims = (width, height)
    print( 'scaling.....')
    scaled_img  = cv2.resize(img, scaled_dims, interpolation = interpolation)
    return scaled_img 

def quantize_image(img_path, 
                   num_clusters, 
                   algo='lloyd', 
                   random_state=0, 
                   max_size_mb=5.8 , 
                   min_resize_factor = 0.1)  : 
    
    img_split= img_path.split('.')
    if len(img_split)>1:
        im_extension      = img_path.split('.')[-1] 
        image_name_prefix = ''.join(cust_split(img_path))[0:-len(im_extension)-1]
        print(image_name_prefix)

    image=cv2.imread(img_path)
    h,w = image.shape[:2]

    image =cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    #reshape
    image =image.reshape(( image.shape[0]*image.shape[1], 3))
    #K-means 
    print(f'# of clusters {clusters}')
    clf=MiniBatchKMeans(n_clusters = clusters, random_state=random_state)
    labels    = clf.fit_predict( image)
    quantized = clf.cluster_centers_.astype("uint8")[ labels]
    #Reshape 

    quantized =quantized.reshape((h, w, 3))
    image = image.reshape( (h, w, 3))
    #Convert from L*a*b to RGB
    quantized = cv2.cvtColor( quantized, cv2.COLOR_LAB2BGR, cv2.CV_8U)
    image =cv2.cvtColor( image, cv2.COLOR_LAB2BGR, cv2.CV_8U)
    ## Saving to disk 
    quantized_image_name = f"{image_name_prefix }_color_quant_{clusters}cs.{im_extension}"
    quantized_file_path = os.path.join(DEFAULT_FILES_PATH,quantized_image_name)
    cv2.imwrite(quantized_file_path, np.hstack( [quantized] ))
    ## resize if needed 
    file_info = os.stat(quantized_file_path)
    quantized_file_size =  file_info.st_size
    quantize_size_mb = quantized_file_size/(1000*1000)
    print(f'quantized size mb : {quantize_size_mb}' )
    if ( quantize_size_mb >= max_size_mb):
        scaling_factor = max_size_mb/quantize_size_mb
        print(f'scaling_factor : {scaling_factor}' )
        if(scaling_factor >= min_resize_factor):
            new_quant_image = resize_image(image_path     = quantized_file_path, 
                                           scaling_factor = scaling_factor
                                           )
            ## save image 
            resized_image_name = quantized_image_name.split('/')[-1]
            new_image_name = f'resized_{resized_image_name}'
            print(os.path.join(DEFAULT_FILES_PATH, new_image_name))
            cv2.imwrite(os.path.join(DEFAULT_FILES_PATH, new_image_name), new_quant_image) 


ap = argparse. ArgumentParser()
ap.add_argument("-process", "--process", required   = True, help="process to run")
ap.add_argument("-image_path", "--image",  required = True, help ="Path to the input image")
ap.add_argument("-clusters", "--clusters", required = False, type = int,  help= "# of clusters")
ap.add_argument("-algo", "--algo", required = False , type = str, help= "kmeans algorithm")
ap.add_argument("-random_state",  "--random_state", required = False , type = int, help= "random_state")
ap.add_argument("-max_ouput_size_mb", "--max_output_size_mb", required=False, type=int,help = "maximum image output size" )

args =vars(ap.parse_args( ) )

if __name__ == "__main__":
    process          =  args.get("process")
    clusters         = args.get("clusters")
    img_path         = args.get("image")
    algo             = args.get("algo")
    random_state     = args.get("random_state")
    max_size_mb      = args.get("max_ouput_size_mb")


    if not algo or algo == None:
        algo = DEFAULT_ALGO

    if not random_state or random_state == None:
        random_state = DEFAULT_RANDOM_STATE

    if not max_size_mb or max_size_mb == None:
        max_size_mb = DEFAULT_MAX_OUTPUT_SIZE_MB

    if process == 'quantize':
        quantize_image(img_path     = img_path, 
                       num_clusters = clusters, 
                       algo         ='lloyd', 
                       random_state = random_state, 
                       max_size_mb  = max_size_mb 
                       ) 

    if process == 'resize_only':
        imageResizeOnly(image_path=img_path)
