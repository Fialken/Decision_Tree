import math
import pandas as pd

def best_entropy(dfx):
    min_entr = math.log2(len(dfx)) # we want the attribute with min entropy to make a split
    best = ""                      # the max val for entropy will be when all values for attrib i1 are different
                                   # let c = len(df)
    for attri in dfx.columns:      # that means max entropy = -c * 1/c * log2(1/c) = - (-log2(c)) = log2(c)
        tmp_entr = entropy(dfx[attri])
        if tmp_entr < min_entr:
            best = attri
            min_entr = tmp_entr
    return best # return the name of the best attribute < lowest entropy means best :D >

def entropy(dfx):
    size = len(dfx)
    ent_val = 0
    for val in dfx.unique():
        count = 0
        for i in range (size):
            if dfx[i] == val:
                count += 1

        p = count/size
        ent_val += (p * math.log2(p))
    return -ent_val


class LeafNode:

    def __init__(self, val):
        classe = val
        counter = 0


class Node:
    
    def __init__(self, attrib, val_ori):
        attribute = attrib
        origin_value = val_ori


class DTree:

    def __init__(self):
        self.__root = None


    def create_DTree(self, dx_train, dy_train):

        #decidir o atributo para a melhor divisao
        best_attrib = best_entropy(dx_train)
        ...