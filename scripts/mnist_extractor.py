import pickle
import gzip
import numpy as np

with gzip.open('mnist.pkl.gz', 'rb') as f:
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

# The original training data has labels as integers. 
# To train the network, we need to convert the integer i into an n-d vector 
#       where the ith index contains 1 and the other entries are 0.
def to_one_hot(y):
    one_hot = np.zeros((len(y), 10), dtype=np.float32)
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot

# Prepare training data (50,000 samples)
train_imgs = training_data[0].astype(np.float32)      # Shape: (50000, 784)
train_labels = to_one_hot(training_data[1])           # Shape: (50000, 10)

# Prepare test data (10,000 samples)
test_imgs = test_data[0].astype(np.float32)            # Shape: (10000, 784)
test_labels = test_data[1].astype(np.int32)            # Shape: (10000,)

# Save to raw binary format
train_imgs.tofile('train_images.bin')
train_labels.tofile('train_labels.bin')
test_imgs.tofile('test_images.bin')
test_labels.tofile('test_labels.bin')