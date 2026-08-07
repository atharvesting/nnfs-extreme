import numpy as np
import matplotlib.pyplot as plt

file_path = "data/output/pixel_data.bin"

pixel_data = np.fromfile(file_path, dtype=np.float32)
reshaped = pixel_data.reshape(28, 28)

plt.imshow(reshaped, cmap="afmhot")
plt.show()