"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
from turtle import width
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if show:
        show_image(img)
    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # Performing enrollment on character images
    keyp,descriptors=enrollment(characters)
 
    # Performing detection of candidate characters in test image and returning coordinates in orignal image and pixel values
    Candidate_Samples,Cordinates=detection(test_img)
    # Performing recognition of candidate characters
    results=recognition(characters,Candidate_Samples,Cordinates,keyp,descriptors)
    return results

def enrollment(characters):
    sift = cv2.SIFT_create()
    
    keypoints=[]   # Stores Key Points of characters
    descriptors=[]  # Stores Descriptors of characters 
    for value, image in characters:
       ret,image = cv2.threshold(image,190,255,0)          # Apply threshold on each character
       each_kp,each_des=sift.detectAndCompute(image,None)  # keypoint and descriptor of each character using sift
       if each_des is not None:                            # if sift returns no descriptors. skip that character
        keypoints.append(each_kp)                          
        descriptors.append(each_des)  
    return (keypoints,descriptors)

def detection(test_img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # This function detects candidates from test_img 
    # Apply threshold on test_img so that characters with lighter black shade are more visible
    T,threshold_img = cv2.threshold(test_img,190,255,0)
    show_image(threshold_img) 
    
    # Get rows and columns in test image
    rows=threshold_img.shape[0]
    cols=threshold_img.shape[1]  
    # keep a copy of image to draw bounded boxes 
    img_copy=threshold_img.copy()    

    detected_img=np.zeros((rows,cols)) #detected_img will contain the labelled pixel values
    #Starting labelling from 1
    label=1    
    
    """Two Pass Algorithm """   
    # 1st pass: labelling Foreground Pixel values
    for i, row in enumerate(threshold_img):
        for j, pixel in enumerate(row):
            if pixel==255:
                continue
            elif pixel==0:
                up=i-1
                left=j-1
                if threshold_img[up][j]!=0 and threshold_img[i][left] != 0:
                    detected_img[i][j]=label
                    label+=1
                elif threshold_img[up][j]==0:
                    detected_img[i][j]=detected_img[up][j]
                elif threshold_img[i][left]==0:
                    detected_img[i][j]=detected_img[i][left]

    # 2nd Pass: Merging Labels which are connected
    for i, row in enumerate(threshold_img):
        for j, pixel in enumerate(row):
            if pixel==255:
                continue
            elif pixel==0:
                up,down,left,right=i-1,i+1,j+1,j-1
                # Checks if up and left neighbours have a label and merges them 
                if detected_img[up][j]!=0 and detected_img[i][left]!=0:
                    min_val=min(detected_img[up][j],detected_img[i][left])
                    detected_img[i][j]=min_val
                    if detected_img[up][j]!=min_val:
                        n=detected_img[up][j]
                        detected_img[detected_img==n]=min_val
                    elif detected_img[i][left]!=min_val:
                         n=detected_img[i][left]
                         detected_img[detected_img==n]=min_val     

                # Checks if down and right neighbours have a label and merges them 

                if detected_img[down][j]!=0 and detected_img[i][right]!=0:
                    min_val=min(detected_img[down][j],detected_img[i][right],detected_img[i][j])
                    detected_img[i][j]=min_val
                    if detected_img[down][j]!=min_val:
                        n=detected_img[down][j]
                        detected_img[detected_img==n]=min_val
                    elif detected_img[i][right]!=min_val:
                         n=detected_img[i][right]
                         detected_img[detected_img==n]=min_val
                                    
                elif detected_img[up][j]!=0 and detected_img[i][left]==0:
                    detected_img[i][j]=detected_img[up][j]
                elif detected_img[i][left]!=0 and detected_img[up][j]==0:
                    detected_img[i][j]=detected_img[i][left]             
                
    #Assign sequential labelling
    for i, label in enumerate(np.unique(detected_img)): 
        x=detected_img.copy()
        x=x.reshape((rows*cols))
        n=label
        get_indexes_value=[]
        for l, m in zip(x,range(len(x))):
            if n==l:
                get_indexes_value.append(m)
        x[get_indexes_value]=i
        detected_img=x.reshape((rows,cols)) 
    
    # Removing bg label            
    labels=np.unique(detected_img)
    labels=np.delete(labels,0)

    # Find Coordinates for each detected label 
    Cordinate_x=[]
    Cordinate_y=[]
    for i in range(len(labels)+1):
        Cordinate_x.append([])
        Cordinate_y.append([])
    
    # Finding occurences of a label
    for i in range(rows):
        for j in range(cols):
            value=detected_img[i][j]
            if value in labels: 
                Cordinate_x[int(value)].append(j)
                Cordinate_y[int(value)].append(i)
    Cordinate_x[0]=[0,0]
    Cordinate_y[0]=[0,0]
    
    Samples_candidates=[]
    Cords_candidates=[]    
       
    
    #Calculating cordinates for each labelled character

    for i in range(len(labels)+1):        
        x_maximum=max(Cordinate_x[i])
        x_minimum=min(Cordinate_x[i])
        y_maximum=max(Cordinate_y[i]) 
        y_minimum=min(Cordinate_y[i])
        x=x_minimum
        y=y_minimum
        height_img=y_maximum-y_minimum
        width_img=x_maximum-x_minimum+1
        candidate_sample=threshold_img[y:y+height_img,x:x+width_img]
        if len(candidate_sample)!=0:
            Samples_candidates.append(np.asarray(candidate_sample))        
            Cords_candidates.append([x,y,width_img,height_img])
            img_copy = cv2.rectangle(img_copy, (x,y),(x+width_img,y+height_img), (0,255,0), 1)

    # Removes Bg label 0
    Cords_candidates.pop(0)
    Samples_candidates.pop(0)
    
    # Add Padding   
    for i in range(len(Samples_candidates)):
        image=Samples_candidates[i]
        top = int(0.2 * image.shape[0]) 
        bottom = top
        left = int(0.2 * image.shape[1])
        right = left
        borderType = cv2.BORDER_CONSTANT
        
        image=cv2.copyMakeBorder(image,top,bottom,left,right,borderType,None,(255,0,0))
        Samples_candidates[i]=image
        
    # Display Result of detection
    show_image(img_copy)
    cv2.imwrite('detected_img.jpg',img_copy)    
    return(Samples_candidates,Cords_candidates)

def recognition(characters,candidates,Cordinates,kp,des):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : This function recognizes the candidates and store the coordinates.
    results=[]      
    # Initiate SIFT detector
    sift = cv2.SIFT_create()     
    
    for i in range(len(candidates)):
        # Get Coordinates for each candidate
        Cord=Cordinates[i]        
        # Duplicating each test sample to find features        
        test=candidates[i]      
        match=[]       
        j=0
        for tag, img in characters:
            if tag=='dot':
                continue
            else:  
                # Compute key points and descriptors for each candidate character in the test image
                upscale=250
                w=int(img.shape[1]*upscale/100)
                h=int(img.shape[0]*upscale/100)
                dim=(w,h)
                img=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
                c=img.shape[0]
                r=img.shape[1]
                dim=(r,c)
                test=cv2.resize(test,dim,interpolation=cv2.INTER_AREA)
                kp_test,des_test = sift.detectAndCompute(test,None)
                if des_test is not None:
                    match_score=[]       
                    for row in des[j]:                    
                        repeated_arr = np.tile(row, (des_test.shape[0], 1))
                        SSD= des_test - repeated_arr
                        SSD=(SSD) ** 2
                        SSD_FINAL=np.sum(SSD, axis=1)
                        SSD_FINAL.sort()
                        if len(SSD_FINAL)>1:
                            Ssd_score=SSD_FINAL[0]/SSD_FINAL[1]
                        else:
                            Ssd_score=SSD_FINAL[0]
                        match_score.append(Ssd_score)
                    count_match=0
                    for value in match_score:
                        if value < 0.62:
                            count_match=count_match+1
                    match.append([tag,count_match])
                j+=1

           
        flag=0
        max_count=0
        for tag, count_match in match:
            if count_match>=4:
                flag=1
                if max_count < count_match:
                    max_count=count_match
                    string=tag
        if flag==0:
            string="UNKNOWN"                       
        # Store recognised test character in the result            
        result={"bbox": Cord ,"name":string}  
        results.append(result)         
    return results

def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])
    
    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)

if __name__ == "__main__":
    main()
