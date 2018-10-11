import numpy as np
import math
import sys
from zellegraphics import *
import time
import matplotlib.pyplot as plt

from SingleLayerPerceptronNetwork import *
from helper import *

def main():
    ins,outs = training_sets_from_file()

    in_real = np.array([1,0,0,1])
    nn = SingleLayerPerceptronNetwork(input_num=len(ins[0]), output_num=len(outs[0]))
    nn.train(ins,outs,graph=True)

    nn.set_input(in_real)
    out = nn.compute()

    nn.win.getMouse()
    nn.win.close()

if __name__ == '__main__':
    main()