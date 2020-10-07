import os
import sys
import cv2 as cv
import numpy as np
from pprint import pprint

processed_data_dir = "/home/insight/Documents/Projects/image-classifier/processed-data"

img_width = 400
img_height = 225
img_dim = img_width * img_height

img_per_game = 50
games_per_genre = 5
genre_count = 2
total_img_count = img_per_game * games_per_genre * genre_count

images = []
labels = []

index_to_genre = {}

print( "Starting to collect data..." )
for i, genre_entry in enumerate( os.scandir( processed_data_dir ) ):

    if not genre_entry.is_dir():
        print( "Warning: A non-dir under processed data dir?" )
        continue

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
np.save( "data.npy", np.hstack( ( images, labels ) ) )  # Append the labels as a column to images, then save it.

