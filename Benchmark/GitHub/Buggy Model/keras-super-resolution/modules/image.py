import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def to_dirname(name):
    """
    ディレクトリ名の"/"有無の違いを吸収する
    # 引数
        name : String, ディレクトリ名
    # 戻り値
        name : String, 変更後
    """
    if name[-1:] == '/':
        return name
    else:
        return name + '/'


def check_dir(name):
    """
    ディレクトリの存在を確認して、存在しなければ作成する
    # 引数
        name : String, ディレクトリ名
    """
    if os.path.isdir(name) == False:
        os.makedirs(name)


def load_image(name, size):
    """
    画像を読み込み配列に格納する
    # 引数
        name : String, 保存場所
        size : List, 画像サイズ
    # 戻り値
        image : Numpy array, 画像データ
    """
    image = Image.open(name)
    image = image.resize((size[0]//2, size[1]//2))
    image = image.resize(size, Image.NEAREST)
    image = np.array(image)
    # 正規化
    image = image / 255
    # モデルの入力次元にあわせる
    image = np.array([image])
    return image


def load_dir(name, size, ext='.jpg'):
    """
    画像群を読み込み配列に格納する
    # 引数
        name : String, 保存場所
        size : List, 画像サイズ
        ext : String, 拡張子
    # 戻り値
        images : Numpy array, 画像データ
    """
    images = []
    for file in tqdm(os.listdir(name)):
        if os.path.splitext(file)[1] != ext:
            # 拡張子が違うなら処理しない
            continue
        image = Image.open(name+file)
        if image.mode != "RGB":
            # 3ch 画像でなければ変換する
            image.convert("RGB")
        # 縮小してから拡大する
        image = image.resize((size[0]//2, size[1]//2))
        image = image.resize(size, Image.NEAREST)
        image = np.array(image)
        images.append(image)
    images = np.array(images)
    # 256階調のデータを0-1の範囲に正規化する
    images = images / 255
    return images


def load_images(name, size, ext='.jpg'):
    """
    画像群を読み込み配列に格納する
    # 引数
        name : String, 保存場所
        size : List, 画像サイズ
        ext : String, 拡張子
    # 戻り値
        x_images : Numpy array, 学習画像データ
        y_images : Numpy array, 正解画像データ
    """
    x_images = []
    y_images = []
    for file in tqdm(os.listdir(name)):
        if os.path.splitext(file)[1] != ext:
            # 拡張子が違うなら処理しない
            continue
        image = Image.open(name+file)
        if image.mode != "RGB":
            # 3ch 画像でなければ変換する
            image.convert("RGB")
        # 縮小してから拡大する
        x_image = image.resize((size[0]//2, size[1]//2))
        x_image = x_image.resize(size, Image.NEAREST)
        x_image = np.array(x_image)
        y_image = image.resize(size)
        y_image = np.array(y_image)
        x_images.append(x_image)
        y_images.append(y_image)
    x_images = np.array(x_images)
    y_images = np.array(y_images)
    # 256階調のデータを0-1の範囲に正規化する
    x_images = x_images / 255
    y_images = y_images / 255
    return x_images, y_images


def save_image(image, name='result.jpg'):
    """
    画像群を任意の場所に保存する
    # 引数
        image : Numpy array, 画像データ
        name : String, ファイル名
    """
    # PILで保存できるように型変換
    image = image[0] * 255
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    # "*.jpg" の形で保存
    image.save(name)


def save_images(images, name, ext='.jpg'):
    """
    画像群を任意の場所に保存する
    # 引数
        images : Numpy array, 画像データ
        name : String, 保存場所
        ext : String, 拡張子
    """
    check_dir(name)
    # PILで保存できるように型変換
    images = images * 255
    images = images.astype(np.uint8)
    for i in range(len(images)):
        image = Image.fromarray(images[i])
        # "*/result[0-9]*.jpg" の形で保存
        image.save(name+'/result'+str(i)+ext)