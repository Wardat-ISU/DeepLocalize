from glob import glob
import cv2
import numpy as np
from skimage import exposure

def parseImagePaths(dir_path, file_extension='.jpg'):
    return glob('{}*{}'.format(dir_path, file_extension))


def expandClass(paths, label, classSize, img_size):
    x = []
    y = []

    for path in paths:
        img = randomizeImg(path, img_size)
        x.append(img)
        y.append(label)

    # Adding 'classSize' amount randomized transformations to the dataset:
    while (len(x) < classSize):
        # Getting a random image and transforming..
        i = np.random.randint(0, len(paths))
        img = randomizeImg(paths[i], img_size)
        x.append(img)
        y.append(label)

    return x, y

def rotateImg(img, angle):
    r = img.shape[0]
    c = img.shape[1]

    img_rotationMatrix = cv2.getRotationMatrix2D((c/2, r/2), angle, 1)

    return cv2.warpAffine(img, img_rotationMatrix, (c, r))


def blurImg(img):
    img = cv2.blur(img, (5, 5))
    return img


def randomizeImg(img_path, img_size):
    img = cv2.imread(img_path)
    theta = np.random.randint(0, 360)

    img = rotateImg(img, theta)
    img = blurImg(img)

    return cv2.resize(img, (img_size, img_size))


def greyscaleImgs(imgs):
    imgs = 0.2989*imgs[:,:,:,0] + 0.5870*imgs[:,:,:,1] + 0.1140*imgs[:,:,:,2]

    return imgs


def normalizeImgs(imgs):
     # source http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
    imgs = (imgs / 255.).astype(np.float32)

    for i in range(imgs.shape[0]):
        imgs[i] = exposure.equalize_hist(imgs[i])

    imgs = imgs.reshape(imgs.shape + (1,))
    return imgs
