from PIL import Image
import random
import time
import math
import json
import copy

random.seed(1)

e = 2.71828


# ACTIVATION BUNDLES

def sigmoid(x):
    try:
        return 1 / (1 + pow(e, -x))
    except OverflowError:
        return 1 if x > 0 else 0


def d_sigmoid(x):
    return x * (1 - x)


tanh = math.tanh


def d_tanh(x):
    return 1 - ((tanh(x)) ** 2)


def leaky_ReLU(inp):
    return inp if inp > 0 else 0.01 * inp


def d_leaky_ReLU(inp):
    return 1 if inp > 0 else 0.01


# UTILITY FUNCTIONS

def flatten(l):
    return sum(l, [])


def lift(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


# TRAINING SETS

def get_ts():
    # print("Getting ts")
    ts = []
    for _ in range(100):
        r1, r2 = random.random(), random.random()
        ts.append([[r1, r2], [1 if r1 > r2 else 0]])
    # print("Done")
    return ts


def check_ts(ts, nn):
    all_errors = []
    r = 0
    for inp, lbl in ts:
        res = nn.forward(inp)
        if lbl[0] == round(res[0]):
            r += 1
    len_ts = len(ts)
    print(f"{r} out of {len_ts} where correct ({r / len_ts * 100}%)!")


def get_mnist():
    print("Getting MNIST")
    with open("mnist_train.csv") as f:
        data = f.readlines()
        s = []
        empty_label = [0] * 10
        for example in data:
            sp = example.split(",")
            example = [int(n) / 255 for n in sp[1:]]
            label = empty_label[:]
            label[int(sp[0])] = 1
            s.append([example, label])
    print("Done with MNIST")
    return s


def get_mnist_without_label():
    print("Getting MNIST")
    with open("mnist_train.csv") as f:
        data = f.readlines()
        s = []
        for example in data:
            sp = example.split(",")
            example = [int(n) / 255 for n in sp[1:]]
            s.append(example)
    print("Done with MNIST")
    return s


def check_mnist(mnist, nn):
    all_errors = []
    r = 0
    for inp, lbl in mnist:
        res = nn.forward(inp)
        if lbl.index(1) == res.index(max(res)):
            r += 1
    len_mnist = len(mnist)
    print(f"{r} out of {len_mnist} where correct ({r / len_mnist * 100}%)!")


# IMAGE CREATION

def turn_to_pic(data=(), type="L", size=(28, 28), name="pic.png"):
    im = Image.new(type, size)
    im.putdata(data)
    im.save(name)
