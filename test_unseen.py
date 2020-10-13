import joblib

from sklearn.metrics import accuracy_score

from load_data import load_data

pipe = joblib.load( "pipe.joblib" )

images, labels = load_data( "unseen-data.npy", 1000 )

print( "Unseen test accuracy: ", accuracy_score( pipe.predict( images ), labels ) )

