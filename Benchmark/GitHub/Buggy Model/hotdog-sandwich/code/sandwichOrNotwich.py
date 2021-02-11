import ImageTools
import pickle
import numpy as np

def main():
    img_size = 128
    classSize = 2000

    # Loading model from .pkl
    model_file = open('model.pkl', 'rb')
    model = pickle.load(model_file)

    # Loading data
    print("\nImporting data..")
    hotdog_files = ImageTools.parseImagePaths('./img/hotdog/')
    print("\t..done.\n")

    # Preprocess the hotdog files, just like what was done in trainModel.py
    # note that the class label isn't necessary, as that is what we're trying to determine.
    print("\nGreyscaling and Normalizing Images..")
    x, _ = ImageTools.expandClass(hotdog_files, 0, classSize, img_size)
    x = np.array(x)
    x = ImageTools.greyscaleImgs(x)
    x = ImageTools.normalizeImgs(x)
    print("\t..done.\n")

    # Generating results from the model:
    results = model.predict(x)
    mean = np.mean(results)
    stddev = np.std(results)

    print("--")
    print("'Is a hotdog a sandwich?''")
    print("RESULTS:")
    print("\tMean: {}".format(mean))
    print("\tStandard Deviation: {}".format(stddev))
    print("--")

main()
