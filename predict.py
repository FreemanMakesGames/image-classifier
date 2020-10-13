import sys
import numpy as np
import cv2 as cv
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

def predict( pipe, img_path ):

    img = cv.imread( img_path, cv.IMREAD_GRAYSCALE )
    img = cv.resize( img, ( 400, 225 ), interpolation = cv.INTER_AREA )
    img = np.array( img, dtype = np.uint8 )
    img = preprocessing.scale( img )
    img = img.reshape( 1, -1 )

    return pipe.predict( img )


pipe = joblib.load( "pipe.joblib" )

print( predict( pipe, sys.argv[ 1 ] ) )

