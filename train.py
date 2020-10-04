import os
import sys
import cv2 as cv
import numpy as np
from pprint import pprint

# Load data from disk, and construct it into a train data object.
data = np.load( "data.npy" )
images = data[ :, 0 : -1 ]
labels = data[ :, -1 ]
data = cv.ml.TrainData_create( images, cv.ml.ROW_SAMPLE, labels )


model = cv.ml.LogisticRegression_create()
model.setLearningRate( 1 )
model.setIterations( 1000 )
model.setTrainMethod( cv.ml.LOGISTIC_REGRESSION_BATCH )

print( "Starting to train the model..." )
model.train( data )

loss, preds = model.calcError( data, True )
print( "Loss: ", loss )
print( "Indices of one's: ", np.where( preds.flatten() == 1 ) )


sys.exit( 0 )


test_img = np.float32( cv.imread( "/home/insight/Documents/Projects/image-classifier/processed-data/third-person-shooter/dead-space/dead-space-0.mp4_0.jpg", cv.IMREAD_GRAYSCALE ) )
test_img = test_img.flatten() / 255
test_img = test_img.reshape( 1, -1 )

retval, results = model.predict( test_img )
print( retval )
print( results )

