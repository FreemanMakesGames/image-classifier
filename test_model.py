import joblib
import argparse

from sklearn.metrics import accuracy_score

from load_data import load_data


argparser = argparse.ArgumentParser( "Test the model on a given data file." )
argparser.add_argument( "--data-file-path", "-dfp" )
args = argparser.parse_args()

pipe = joblib.load( "pipe.joblib" )

images, labels = load_data( args.data_file_path, 1000 )

print( "Accuracy: ", accuracy_score( pipe.predict( images ), labels ) )

