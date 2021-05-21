import numpy as np
import xlrd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
from scipy.optimize import minimize

workbook = xlrd.open_workbook("Ex8data.xlsx")
worksheet = workbook.sheet_by_index(0)

X = []
for i in range(1,231):
    Xi = []
    for j in range(0,3):
        Xi.append(worksheet.cell_value(i,j))
    X.append(Xi)

Y = []
for i in range(1,231):
    Y.append(worksheet.cell_value(i,3))

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=10)

nn1 = MLPRegressor(hidden_layer_sizes = (11,3),activation='tanh',max_iter = 1000,random_state =10,solver='lbfgs').fit(X_train,Y_train)
nn2 = MLPRegressor(hidden_layer_sizes = (11,8),activation='tanh',max_iter = 1000,random_state =10,solver='lbfgs').fit(X_train,Y_train)
nn3 = MLPRegressor(hidden_layer_sizes = (19,3),activation='tanh',max_iter = 1000,random_state =10,solver='lbfgs').fit(X_train,Y_train)

Neural_Nets = []
Neural_Nets.append(nn1)
Neural_Nets.append(nn2)
Neural_Nets.append(nn3)

for i in range(0,3):
    nn = Neural_Nets[i]
    def func(X):
        Ypred = nn.predict([X])
        return(-1*Ypred[0])
    print("neural network : Hidden layer1:",Neural_Nets[i].hidden_layer_sizes[0],"Hidden layer2 :",Neural_Nets[i].hidden_layer_sizes[1])    
    Bounds = [(0.0,1.0),(0.0,1.0),(0.0,1.0)]
    Max = minimize(func,[0.0,0,0],bounds =Bounds)
    print("Maximum predicted efficiency value:",-1*Max.fun)
    print("Optimal solutions:",[Max.x[0],Max.x[1],Max.x[2]],'\n')

print("Maximum efficiency from the data:",max(Y))



