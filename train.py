import os
import sys
import numpy as np
from pprint import pprint

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from disk, and construct it into a train data object.
data = np.load( "data.npy" )
images = data[ :, 0 : -1 ]
labels = data[ :, -1 ]

pipe = make_pipeline( LogisticRegression( random_state = 0 ) )
X_train, X_test, y_train, y_test = train_test_split( images, labels, random_state = 0 )

print( "Training started." )
pipe.fit( X_train, y_train )

print( "Accuracy score: ", accuracy_score( pipe.predict( X_test ), y_test ) )

sys.exit( 0 )

def test( img_path ):
    test_img = np.float32( cv.imread( img_path, cv.IMREAD_GRAYSCALE ) )
    test_img = test_img.flatten() / 255
    test_img = test_img.reshape( 1, -1 )

    retval, results = model.predict( test_img )
    print( retval )

test( "processed-data/first-person-shooter/csgo/csgo-0.mp4_10.jpg" )
test( "processed-data/first-person-shooter/pubg/pubg-0.mp4_32.jpg" )
test( "processed-data/third-person-shooter/resident-evil-3-remake/resident-evil-3-remake-0.mp4_13.jpg" )
test( "processed-data/third-person-shooter/sniper-elite-4/sniper-elite-4-0.mp4_42.jpg" )

