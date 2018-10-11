import numpy as np
from zellegraphics import *
import math
import matplotlib.pyplot as plt

class SingleLayerPerceptronNetwork():

    def __init__(self, input_num, output_num):
        self.input_num = input_num
        self.output_num = output_num
        self.weights = np.random.rand(input_num, 1)

        self.inputs = []
        self.outputs = []

        self.win = GraphWin("Perceptron Neural Network", 500, 500)
        self.graphic_input_neurons = []
        self.graphic_input_labels = []
        self.graphic_weight_labels = []
        self.graphic_output_labels = []

        self.init_display()

    def set_input(self,input):
        self.inputs = input

    def set_output(self,output):
        self.outputs = output

    def update_display(self):
        for i in range(self.input_num):
            input_text = self.graphic_input_labels[i]
            input_text.setText(str(self.inputs[i]))

            weight_text = self.graphic_weight_labels[i]
            weight_text.setText(str(self.weights[i]))

        out_text = self.graphic_output_labels[0]
        out_text.setText(str(self.outputs[0]))



    def init_display(self):
        start_point = (50,50)
        r = 20
        for i in range(0, self.input_num):
            c = Circle(Point(start_point[0],start_point[1]+(i*50)),r)
            self.graphic_input_neurons.append(c)
            c.draw(self.win)

            input_text = Text(Point(start_point[0],start_point[1]+(i*50)),"")
            self.graphic_input_labels.append(input_text)
            input_text.draw(self.win)


            weight_text = Text(Point(start_point[0]+50,start_point[1]+(i*50)-20),"")
            weight_text.setSize(7)
            self.graphic_weight_labels.append(weight_text)
            weight_text.draw(self.win)

            start = Point(start_point[0]+r,start_point[1]+(i*50))
            end = Point(start_point[0] + 100, ((self.input_num + 1) * 50) / 2)
            Line(start, end).draw(self.win)

            if(i==0):
                Circle(Point(end.x+r,end.y),r).draw(self.win)
                neuron_text = Text(Point(end.x+r,end.y),"N")
                neuron_text.draw(self.win)

                Line(Point(end.x+2*r,end.y),Point(end.x+100-r,end.y)).draw(self.win)

                out_circle = Circle(Point(end.x+100,end.y),r).draw(self.win)
                out_text = Text(Point(end.x+100,end.y),"")
                out_text.setSize(7)
                self.graphic_output_labels.append(out_text)
                out_text.draw(self.win)





    def sigmoid(self, x, deriv=False):
        if not deriv:
            return (1)/(1+math.pow(math.e,-x))
        else:
            return x * (1-x)

    def train(self,ins,outs,graph=False):
        errors = []
        iters = 1000
        plt.title("Percent Error, per iteration and test case")
        plt.xlabel("Iteration")
        plt.ylabel("% Error")
        for x in range(0, iters):
            error_sum = 0
            colors = ['red','blue','green','orange']
            for i in range(0, len(ins)):
                self.set_input(ins[i])
                self.set_output(outs[i])
                train_out,error = self.back_prop_train()
                if(graph):
                    plt.scatter(x,error,color=colors[i])
                    plt.pause(0.005)

            errors.append(error_sum/len(ins))

    def back_prop_train(self):
        self.update_display()
        if self.inputs.shape[0] == self.input_num and self.outputs.shape[0] == self.output_num:
            #(x1 * w1) + (x2 * w2)...
            sum = np.dot(self.inputs,self.weights)
            sig = self.sigmoid(sum)
            error = self.outputs[0] - sig
            for x in range(self.input_num):
                self.weights[x] += error * self.inputs[x] * self.sigmoid(sig,True)
            print("training...", self.inputs, self.outputs, " | ", sig)
            return sig,error
        else:
            sys.stderr.write("Input/Output array incorrect shape!\n")
            return None

    def compute(self):
        if(self.inputs.shape[0] == self.input_num):
            # (x1 * w1) + (x2 * w2)...
            sum = np.dot(self.inputs, self.weights)
            sig = self.sigmoid(sum)
            self.outputs = [sig]
            self.update_display()
            print("computing...",self.inputs,sig)
            return sig
        else:
            sys.stderr.write("Input/Output array incorrect shape!\n")
            return None