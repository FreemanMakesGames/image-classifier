import numpy as np

from sklearn import preprocessing


def load_data( data_file_path, max_rows ):

    # Load data from disk, and construct it into a train data object.
    data = np.load( data_file_path )

    # Use less data due to insufficient memory.
    np.random.shuffle( data )
    if data.shape[ 0 ] > max_rows:
        data = data[ 0 : max_rows, : ]

    images = data[ :, 0 : -1 ]
    labels = data[ :, -1 ]

    # Preprocessing. This increases accuracy.
    images = preprocessing.scale( images )

    return images, labels

