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

#params_search = RandomizedSearchCV( pipe, { "logistic__C": np.logspace( -2, 2, 10 ) } )  # Best C is 0.0278, resulting in test acc 0.864.
pipe = Pipeline( steps = [ ( "logistic", LogisticRegression( max_iter = 300, C = 0.0278, random_state = 0 ) ) ] )

X_train, X_test, y_train, y_test = train_test_split( images, labels, random_state = 0 )

print( ">>> Training started." )
#params_search.fit( X_train, y_train )
pipe.fit( X_train, y_train )

#print( "Searched best parameters:\n", pformat( params_search.best_params_, indent = 4 ) )
print( "Pipe parameters:\n", pformat( pipe.get_params(), indent = 4 ) )
print( "Training accuracy: ", accuracy_score( pipe.predict( X_train ), y_train ) )
print( "Test accuracy: ", accuracy_score( pipe.predict( X_test ), y_test ) )


def test( img_path ):
    test_img = np.float32( cv.imread( img_path, cv.IMREAD_GRAYSCALE ) )
    test_img = test_img.flatten() / 255
    test_img = test_img.reshape( 1, -1 )

    retval, results = model.predict( test_img )
    print( retval )

