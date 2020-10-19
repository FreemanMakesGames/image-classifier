import os
import sys
import numpy as np
import cv2 as cv
import json
import joblib
import argparse
from pprint import pprint
from pprint import pformat

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

from load_data import load_data


argparser = argparse.ArgumentParser( description = "Start training." )
argparser.add_argument( "--X-rows", "-Xr", type = int, help = "Number of examples to read from data.npy" )
argparser.add_argument( "--max-iter", "-mi", type = int )
argparser.add_argument( "-C", type = float )
argparser.add_argument( "--search-params", "-sp", action = "store_true", help = "Grid search for parameters?" )
args = argparser.parse_args()
X_rows = args.X_rows
max_iter = args.max_iter
C = args.C


images, labels = load_data( "data/training/data.npy", X_rows )

pipe = Pipeline( steps = [ ( "logistic", LogisticRegression( max_iter = max_iter, C = C, random_state = 0 ) ) ] )
params_search = GridSearchCV( pipe, { "logistic__C": np.logspace( -2, 2, 10 ) } )  # C = 0.01 seems okay.

if args.search_params:
    estimator = params_search
else:
    estimator = pipe

X_train, X_test_same_games, y_train, y_test_same_games = train_test_split( images, labels, random_state = 0 )
X_test_new_games, y_test_new_games = load_data( "data/training-test/data.npy", 3000 )

print( ">>> Training started." )
estimator.fit( X_train, y_train )

print( "Estimator parameters:\n", pformat( estimator.get_params(), indent = 4 ) )
print( "Training accuracy: ", accuracy_score( estimator.predict( X_train ), y_train ) )
print( "Same games test accuracy: ", accuracy_score( estimator.predict( X_test_same_games ), y_test_same_games ) )
print( "New games test accuracy: ", accuracy_score( estimator.predict( X_test_new_games ), y_test_new_games ) )


joblib.dump( estimator, "estimator.joblib" )


# Visualize weights.
cv.imwrite( "weights.jpg", estimator[ "logistic" ].coef_.reshape( 225, 400 ) * 2550 * 20 )

