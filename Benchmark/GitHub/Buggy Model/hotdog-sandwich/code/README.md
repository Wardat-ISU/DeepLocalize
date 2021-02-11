# hotdog-sandwich
## By: Sammy Haq

### Dataset Links:
- Sandwich dataset: https://www.kaggle.com/brtknr/sushisandwich/version/2
- Hotdog dataset: http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07697537
- Food dataset: http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00021265

### Code Dependencies
- *Keras.* Prior to this course, I was most proficient in Keras and the individuality and free reign on this project allowed me to go back to the architecture. (Also requires its dependences, like TensorFlow.)
- *OpenCV.*
- *Scikit-Image.*
- *Scikit-Learn.*
- *numpy.*

### Code Structure
- *getImages.py* is what is used to grab all the images from their respective image-net synsets. The sandwich dataset from Kaggle is to be manually downloaded.
- *ImageTools.py* is a bunch of image manipulation methods I put in a corner so the rest of the code stayed clean.
- *model.pkl* is the saved model from trainModel.py.
- *README.md* is where you are right now.
- *sandwichOrNotwitch.py* is the program that loads the model (model.pkl), processes the hot dog images in img/hotdog/, and reports back the mean and standard deviation of all of the hot dog images.
- *trainModel.py* contains the structure of the model, a reference to the GitHub of the CNN the model is based off of, and the train code of the CNN. This is also where the preprocessing of the sandwich and food images take place, number of affine transforms to do to simulate more data, etc.
