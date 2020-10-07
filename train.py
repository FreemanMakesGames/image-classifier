import os
import sys
import numpy as np
import json
from pprint import pprint
from pprint import pformat

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

# Load data from disk, and construct it into a train data object.
data = np.load( "data.npy" )
images = data[ :, 0 : -1 ]
labels = data[ :, -1 ]

# Preprocessing. This increases accuracy.
images = preprocessing.scale( images )

pipe = Pipeline( steps = [ ( "logistic", LogisticRegression( max_iter = 300, C = 0.0278, random_state = 0 ) ) ] )
#params_search = RandomizedSearchCV( pipe, { "logistic__C": np.logspace( -2, 2, 10 ) } )  # Best C is 0.0278, resulting in test acc 0.864.

X_train, X_test, y_train, y_test = train_test_split( images, labels, random_state = 0 )

train_sizes, train_scores, valid_scores = learning_curve( pipe, X_train, y_train, cv = 5 )

pprint( train_scores )
pprint( valid_scores )


sys.exit( 0 )


print( ">>> Training started." )
#params_search.fit( X_train, y_train )
pipe.fit( X_train, y_train )

print( "Pipe parameters:\n", pformat( pipe.get_params(), indent = 4 ) )
#print( "Searched best parameters:\n", pformat( params_search.best_params_, indent = 4 ) )
print( "Training accuracy: ", accuracy_score( params_search.predict( X_train ), y_train ) )
print( "Test accuracy: ", accuracy_score( params_search.predict( X_test ), y_test ) )

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

