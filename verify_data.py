# Reshape first and last array from the data.npy of training and testing data sets, into images, to make sure the labeling of training and testing match.

import cv2 as cv
import numpy as np

training = np.load( "data/training/data.npy" )

training_first = training [ 0 ][ 0 : -1 ].reshape( 225, 400 )
cv.imwrite( "training_first.jpg", training_first )

training_last = training[ -1 ][ 0 : -1 ].reshape( 225, 400 )
cv.imwrite( "training_last.jpg", training_last )

test = np.load( "data/params-search-test/data.npy" )

test_first = test[ 0 ][ 0 : -1 ].reshape( 225, 400 )
cv.imwrite( "test_first.jpg", test_first )

test_last = test[ -1 ][ 0 : -1 ].reshape( 225, 400 )
cv.imwrite( "test_last.jpg", test_last )

