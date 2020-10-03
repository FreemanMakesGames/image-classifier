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

model = cv.ml.LogisticRegression_create()
model.setLearningRate( 0.001 )
model.setIterations( 10 )
model.setTrainMethod( cv.ml.LOGISTIC_REGRESSION_BATCH )

images = []
labels = []

index_to_genre = {}

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

            # Read and prepare image.
            img = cv.imread( img_entry.path, cv.IMREAD_GRAYSCALE )
            img = np.float32( img.flatten() / 255 )

            images.append( img )
            labels.append( i )

# Create train data object.
images = np.array( images )
labels = np.float32( np.array( labels ) )
#data = cv.ml.TrainData_create( images, cv.ml.ROW_SAMPLE, labels )

pprint( labels )

model.train( images, cv.ml.ROW_SAMPLE, labels )

pprint( "Thetas: ", model.get_learnt_thetas() )

test_img = np.float32( cv.imread( "/home/insight/Documents/Projects/image-classifier/processed-data/third-person-shooter/dead-space/dead-space-0.mp4_0.jpg", cv.IMREAD_GRAYSCALE ) )
test_img = test_img.flatten() / 255
test_img = test_img.reshape( 1, -1 )

retval, results = model.predict( test_img )
print( retval )
print( results )

