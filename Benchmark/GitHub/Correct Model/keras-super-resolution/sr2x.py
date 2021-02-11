import os
from argparse import ArgumentParser
from modules.file import load_model
from modules.image import load_image, load_dir, save_image, save_images, to_dirname
from modules.interface import show, get_input


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default=None)
    parser.add_argument('-d', '--dir', type=str, default=None)
    parser.add_argument('-o', '--output', type=str, default=None)
    return parser.parse_args()


def single(model, name, out, size):
    if os.path.isfile(name) == False:
        print('File not exist')
        return
    image = load_image(name=name, size=size)
    prediction = model.predict(image)
    save_image(image=prediction, name=out)


def bundle(model, name, out, size):
    if os.path.isdir(name) == False:
        print('Directory not exist')
        return
    # load_imagesという名前が使われてたから仕方ないんや
    images = load_dir(name=name, size=size)
    prediction = model.predict(images, verbose=1)
    save_images(images=prediction, name=out)


def continuous(model, size):
    print('Enter the file name (*.jpg)')
    while True:
        # 標準入力から取得
        value = get_input()
        # ファイルの存在確認
        if os.path.isfile(value) == False:
            print('File not exist')
            continue
        image = load_image(name=value, size=size)
        show(image)
        prediction = model.predict(image)
        show(prediction)


def main():
    args = get_args()
    if args.file:
        filename = args.file
    else:
        filename = False
    if args.dir:
        dirname = args.dir
    else:
        dirname = False
    if args.output:
        outname = args.output
    else:
        outname = False
    model = load_model('model.json')
    model.load_weights('weights.hdf5')
    # モデルから画像サイズを取得
    size = (model.input_shape[1], model.input_shape[2])
    if filename and outname:
        single(model=model, name=filename, out=outname, size=size)
    elif filename and not outname:
        print("error")
        exit()
    if dirname and outname:
        dirname = to_dirname(dirname)
        outname = to_dirname(outname)
        bundle(model=model, name=dirname, out=outname, size=size)
    elif dirname and not outname:
        print("error")
        exit()
    if not filename and not dirname:
        continuous(model=model, size=size)


if __name__ == '__main__':
    main()
