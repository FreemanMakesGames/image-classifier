import os
import sys
import numpy as np
import cv2 as cv
import json
import joblib
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

# Use less data due to insufficient memory.
np.random.shuffle( data )
data = data[ 0 : 3000, : ]

images = data[ :, 0 : -1 ]
labels = data[ :, -1 ]

# Preprocessing. This increases accuracy.
images = preprocessing.scale( images )

pipe = Pipeline( steps = [ ( "logistic", LogisticRegression( max_iter = 3000, C = 0.01, random_state = 0 ) ) ] )
#params_search = RandomizedSearchCV( pipe, { "logistic__C": np.logspace( -2, 2, 10 ) } )  # Best C is 0.01, resulting in test acc 0.896.

X_train, X_test, y_train, y_test = train_test_split( images, labels, random_state = 0 )

print( ">>> Training started." )
#params_search.fit( X_train, y_train )
pipe.fit( X_train, y_train )

#print( "Searched best parameters:\n", pformat( params_search.best_params_, indent = 4 ) )
print( "Pipe parameters:\n", pformat( pipe.get_params(), indent = 4 ) )
#print( "Training accuracy: ", accuracy_score( params_search.predict( X_train ), y_train ) )
print( "Training accuracy: ", accuracy_score( pipe.predict( X_train ), y_train ) )
#print( "Test accuracy: ", accuracy_score( params_search.predict( X_test ), y_test ) )
print( "Test accuracy: ", accuracy_score( pipe.predict( X_test ), y_test ) )


joblib.dump( pipe, "pipe.joblib" )


# Visualize weights.
cv.imwrite( "weights.jpg", pipe[ "logistic" ].coef_.reshape( 225, 400 ) * 2550 * 2 )

