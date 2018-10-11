import numpy as np

def training_sets_from_file():
    f = open("data/data1.txt","r")
    inputs = []
    outputs = []
    for l in f.readlines():
        l = l.split(" ")
        ins = l[0].strip()
        outs = l[1].strip()
        ins = list(map(int, ins.split(",")))
        outs = list(map(int,outs.split(",")))

        inputs.append(ins)
        outputs.append(outs)
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    return (inputs,outputs)