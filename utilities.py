#
# KTH Royal Institute of Technology
# DD2424: Deep Learning in Data Science
# Assignment 3
#
# Carlo Rapisarda (carlora@kth.se)
#

import os

def unpickle(filename):
    import pickle
    with open(filename, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

def play_bell():
    os.system('afplay /System/Library/Sounds/Ping.aiff')
