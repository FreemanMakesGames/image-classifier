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


# Parse args.
argparser = argparse.ArgumentParser( description = "Start training." )
argparser.add_argument( "--X-rows", "-Xr", type = int, help = "Number of examples to read from data.npy" )
argparser.add_argument( "--max-iter", "-mi", type = int )
argparser.add_argument( "-C", type = float )
argparser.add_argument( "--search-params", "-sp", action = "store_true", help = "Grid search for parameters?" )
args = argparser.parse_args()
X_rows = args.X_rows
max_iter = args.max_iter


# Load and split data sets.
images, labels = load_data( "data/training/data.npy", X_rows )
X_train, X_test_same_games, y_train, y_test_same_games = train_test_split( images, labels, random_state = 0 )
# Use data from unseen games for scoring params search.
X_test_params_search, y_test_params_search = load_data( "data/params-search-test/data.npy", 3000 )
# TODO: Load final test.

best_pipe = None
if args.search_params:
    best_score = 0.0
    # Params search with manual for-loops, because the scoring is using a different data set.
    for C in np.logspace( -3, 1, 10 ):
        pipe = Pipeline( steps = [ ( "logistic", LogisticRegression( max_iter = max_iter, C = C, random_state = 0 ) ) ] )
        print( f">>> Training started for pipe candidate with C = {C}." )
        pipe.fit( X_train, y_train )
        score = accuracy_score( pipe.predict( X_test_params_search ), y_test_params_search )
        if score > best_score:
            best_pipe = pipe
            best_score = score
else:
    best_pipe = Pipeline( steps = [ ( "logistic", LogisticRegression( max_iter = max_iter, C = args.C, random_state = 0 ) ) ] )
    print( ">>> Training started without params search." )
    best_pipe.fit( X_train, y_train )


print( "Estimator parameters:\n", pformat( best_pipe.get_params(), indent = 4 ) )
print( "Training accuracy: ", accuracy_score( best_pipe.predict( X_train ), y_train ) )
print( "Same games test accuracy: ", accuracy_score( best_pipe.predict( X_test_same_games ), y_test_same_games ) )
print( "Params search test accuracy: ", accuracy_score( best_pipe.predict( X_test_params_search ), y_test_params_search ) )


joblib.dump( best_pipe, "best_pipe.joblib" )

