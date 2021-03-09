import numpy as np
from pandas import DataFrame as df
import matplotlib.pyplot as plt
# %matplotlib inline

gain = 10
offset_x= 0.2
offset_green = 0.6

def sigmoid(x, gain=1, offset_x=0):
    return ((np.tanh(((x+offset_x)*gain)/2)+1)/2)

def colorBarRGB(x):
    x = (x * 2) - 1
    red = sigmoid(x, gain, -1*offset_x)
    blue = 1-sigmoid(x, gain, offset_x)
    green = sigmoid(x, gain, offset_green) + (1-sigmoid(x,gain,-1*offset_green))
    green = green - 1.0
    return (blue,green,red)

#入力値は0.0〜1.0の範囲
data = [colorBarRGB(x*0.001) for x in range(0,1000)]

color = df(data)
color.plot()
color.show()
