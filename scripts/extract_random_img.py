import pickle
import gzip
import random
import numpy as np

with gzip.open("mnist.pkl.gz", "rb") as f:
    training_data, _, _ = pickle.load(f, encoding="latin1")

sample_index = random.randrange(len(training_data[0]))
img = training_data[0].astype(np.float32)[sample_index]

output_path = "img.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("std::vector<float> img = {")
    f.write(", ".join(f"{num:g}" for num in img))
    f.write("}\n")