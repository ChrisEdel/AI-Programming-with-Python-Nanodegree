#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: Christian Edelmayer
# DATE CREATED: 10.01.2022                
# REVISED DATE: 
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """
    # Replace None with the results_dic dictionary that you created with this
    # function
    filename_list = listdir(image_dir)
    
    results_dic = dict()
    
    for filename in filename_list:
        if filename not in results_dic:
            # Sets pet_image variable to a filename 
            pet_image = filename

            # Sets string to lower case letters
            low_pet_image = pet_image.lower()

            # Splits lower case string by _ to break into words 
            word_list_pet_image = low_pet_image.split("_")

            # Create pet_name starting as empty string
            pet_name = ""

            # Loops to check if word in pet name is only
            # alphabetic characters - if true append word
            # to pet_name separated by trailing space 
            for word in word_list_pet_image:
                if word.isalpha():
                    pet_name += word + " "

            # Strip off starting/trailing whitespace characters 
            pet_name = pet_name.strip()
            
            results_dic[filename] = [pet_name]
        else:
            print("** Warning: Key=", filename, 
                "already exists in results_dic with value =", 
                results_dic[filename])

    return results_dic
