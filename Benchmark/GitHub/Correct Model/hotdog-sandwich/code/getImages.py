import urllib
import urllib.request
import cv2
import os
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("src", help="Image synset URL.")
parser.add_argument("dest", help="Destination Folder.")

def main():

    # link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07697537'
    # dest = 'img/hotdog'
    args = parser.parse_args()
    link = args.src
    dest = args.dest

    pic_num = 1

    if not os.path.exists(dest):
        os.makedirs(dest)

    imURLs = str(urllib.request.urlopen(link).read())

    for imURL in imURLs.split('\\n'):
        try:
            print("Attempting to retreive " + imURL)
            urllib.request.urlretrieve(imURL, dest+"/"+str(pic_num)+".jpg")
            img = cv2.imread(dest+"/"+str(pic_num)+".jpg")

            # Do preprocessing if you want
            if img is not None:
                # do more stuff here if you want
                cv2.imwrite(dest+"/"+str(pic_num)+".jpg",img)
                pic_num += 1

        except Exception as e:
            print(str(e))

        except KeyboardInterrupt as oe:
            print("\tRetreival of {} aborted.".format(imURL))
            continue

main()
