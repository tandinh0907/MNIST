import numpy as np

def dataToArr(filename, has_label=0):
    arr = np.loadtxt(filename, delimiter = ",", skiprows = 1).astype (np.uint8)
    label = np.zeros((arr.shape[0]))
    if (has_label == 1):
        label = arr[:,0].astype (np.uint8)
        arr = arr[:,1:]
    arr = np.reshape (arr/255, (arr.shape[0], 1, 28, 28))
    return arr, label

def outputFile(filename, arr):
    np.savetxt(filename, arr, delimiter = ',', fmt = '%1d', header = "ImageId,Label", comments='')