import numpy as np
import pandas as pd


def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

x_min, x_max = 0, 1000
y_min, y_max = 0, 1000
step = 1


x_values = np.arange(x_min, x_max + step, step)
y_values = np.arange(y_min, y_max + step, step)
X, Y = np.meshgrid(x_values, y_values)


Z = f(X, Y)


data = {
    'X': X.flatten(),  
    'Y': Y.flatten(),  
    'Z': Z.flatten()   
}


df = pd.DataFrame(data)


output_file = 'output_data.xlsx'
df.to_excel(output_file, index=False)

