from os import listdir, path, errno, makedirs
from shutil import copyfile

import re
from PIL import Image, ImageStat

import numpy as np
import pandas as pd
from scipy import stats
import scipy.fftpack
import matplotlib.pyplot as plt
import cv2 as cv
import scipy.misc


def data_gen(data_root='data/'):
    _figs_paths = filter(lambda p: re.match(r'figs_\d', p), sorted(listdir(data_root)))
    _figs_paths = map(lambda p: path.join(data_root, p), _figs_paths)

    for figs_path in _figs_paths:
        names = filter(lambda p: re.match(r'\w\d{4}_\d{2}\.\w{3}', p), listdir(figs_path))
        names = map(lambda p: p.split('.')[0], names)

        mem = set()
        for f in names:
            if f in mem:
                continue
            mem.add(f)

            name_path = path.join(figs_path, f) + '.{}'
            img_path = name_path.format('png')
            cls_path = name_path.format('txt')

            cls = dict()
            with open(cls_path, 'r') as f:
                for line in f:
                    attr, value = map(str.strip, line.lower().split(':', 1))
                    cls[attr] = value

            yield name_path, cv.imread(img_path, 0), cls


def get_load_data_func(ata_root='data/'):
    gen = data_gen(data_root)

    def wrapper(n):
        r = []

        for i in range(n):
            try:
                r.append(next(gen))
            except StopIteration:
                return r

        return r

    return wrapper


def gen_by_class(cls, cls_group='cls_group'):
    file_path = path.join(cls_group, 'cls_{}'.format(cls))
    for f in listdir(file_path):
        f = path.join(file_path, f)

        yield cv.imread(f, 0)


def get_load_data_by_class_func(cls, data_root='cls_group'):
    gen = gen_by_class(cls, data_root)

    def wrapper(n):
        r = []

        for i in range(n):
            try:
                r.append(next(gen))
            except StopIteration:
                return r

        return r

    return wrapper


def group_file_by_class(data_root='data/'):
    df = pd.DataFrame(
        list(map(lambda t: (t[0].format('png'), t[2]['class']), util.data_gen(data_root))),
        columns='file_path cls'.split()
    )

    for c in df.cls.unique():
        d = 'cls_group/cls_{}'.format(c)
        try:
            makedirs(d)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        files = df[df.cls == c].file_path
        for f in files:
            copyfile(f, path.join(d, f.split('/')[-1]))


def normalize(img, M0=250, VAR0=100):
    I = img.copy()
    M = I.sum() / I.shape[0] / I.shape[1]

    VAR = ((I - M) ** 2).sum() / I.shape[0] / I.shape[1]
    G = -np.sqrt(VAR0 * (I - M) ** 2 / VAR)
    G[I > M] *= -1
    G += M0
    return G


def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi/32):
        kern = cv.getGaborKernel((ksize, ksize), 5, theta, 10.0,1, 0, ktype=cv.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv.filter2D(img, cv.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


def to_timeseries(img, n_scale=2000, reshape=lambda t: t.reshape(-1)):
    filters = build_filters()
    res = process(normalize(img), filters)
    mean = np.mean(res) * 1.2
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if (res[i][j] < mean):
                res[i][j] = 0
            else:
                res[i][j] = 255

    # scipy.misc.imsave('outfile.jpg', res)

    img = res.copy()
    skel = res.copy()
    skel[:,:] = 0
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    while True:
        eroded = cv.morphologyEx(img, cv.MORPH_ERODE, kernel)
        temp = cv.morphologyEx(eroded, cv.MORPH_DILATE, kernel)
        temp  = cv.subtract(img, temp)
        skel = cv.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv.countNonZero(img) == 0:
            break

    # before skeletionization
    preFFT = t(res)

    # Number of samplepoints
    N = preFFT.size // n_scale

    # sample spacing
    T = 1.0 / (255 * 2)
    y = preFFT
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
    return xf[1:], (2.0 / N * np.abs(yf[:N // 2]))[1:]
