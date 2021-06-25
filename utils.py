import numpy as np
import torch
from torch.distributions.categorical import Categorical

def ToTensor(x):

    return torch.from_numpy(x).type(torch.float)


class Struct(object):

    def __init__(self): 

        pass

def cartcheckleaf(tree,s , feature_data):

    if tree.type == 0:
        leaf = tree.index
        val = tree.mean

        return leaf, val
    else:

        if feature_data.splittable[s,tree.test - 1] == 0:
            branch = tree.ltTree
        else:
            branch = tree.gtTree

        leaf, val = cartcheckleaf(branch,s,feature_data)

        return leaf, val

def cartaverage(tree,feature_data):

    if tree.type == 0:
        temp = np.shape(feature_data.splittable)[0]
        R = np.tile(tree.mean, (temp, 1))

        return R
    else:
        ltR = cartaverage(tree.ltTree,feature_data)
        gtR = cartaverage(tree.gtTree,feature_data)


        temp = np.shape(ltR)[1]
        temp1 = np.expand_dims(feature_data.splittable[:,tree.test - 1], axis=1)
        ind = np.tile(temp1, (1, temp))
        R = (1-ind)*ltR + ind*gtR
        return R

def maxentsoftmax(q):


    maxx = np.expand_dims(q.max(axis=1), axis=1)

    temp1 = np.shape(q)[1]
    temp2 = np.tile(maxx, (1, temp1))
    temp3 = np.exp(q - temp2)
    temp4 = np.sum(temp3,axis=1)
    temp5 = np.expand_dims(np.log(temp4), axis=1)
    v = maxx + temp5

    return v
