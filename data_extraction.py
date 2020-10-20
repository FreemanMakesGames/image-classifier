import os
import sys
import cv2 as cv
import numpy as np


def get_data_from_images( img_dir, save_path ):

    images = []
    labels = []

    index_to_genre = {}

    print( "Starting to extract data..." )
    for i, genre_entry in enumerate( os.scandir( img_dir ) ):

        if not genre_entry.is_dir():
            print( "Warning: A non-dir under processed data dir?" )
            continue

########Skip third-person shooters for now#################
        if genre_entry.name == "third-person-shooter":
            continue
###########################################################

        genre_name = genre_entry.name
        index_to_genre[ i ] = genre_name

        for game_entry in os.scandir( genre_entry.path ):

            # Skip info.json and so on.
            if not game_entry.is_dir():
                continue

            for img_entry in os.scandir( game_entry.path ):

                name, ext = os.path.splitext( img_entry.name )

                if ext != ".jpg":
                    print( "Warning: A non-jpg under a game dir?" )
                    continue

                img = cv.imread( img_entry.path, cv.IMREAD_GRAYSCALE ).flatten()

                images.append( img )
                labels.append( i )

    # Make collected data into numpy matrix and save it.
    images = np.array( images, dtype = np.uint8 )
    labels = np.array( labels, dtype = np.uint8 ).reshape( -1, 1 )
    np.save( save_path, np.hstack( ( images, labels ) ) )  # Append the labels as a column to images, then save it.

