# printing some statements throughout the process, because the output is taking time to be produced
# So, it doesn't feel like that it stuck in an infinite loop or its not working.
print("--Importing usefull libraries--")
from scipy.sparse.construct import rand
import xlrd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
print("--imported usefull libraries--")



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

print("--data preprocessing done --")
print("--neural networks fitting started--")
Neural_Nets = []
for i in range(1,21):
    for j in range(1,21):
        nn = MLPRegressor(hidden_layer_sizes=(i,j),activation = 'tanh',max_iter = 750,random_state=10,solver= 'lbfgs').fit(X_train,Y_train)
        Neural_Nets.append(nn)
print("--neural network fitting ended--")
mape = []
for i in range(0,400):
    nn = Neural_Nets[i]
    Y_pred = nn.predict(X_test)
    mape.append(mean_absolute_percentage_error(Y_test,Y_pred))

minI = []
minM = []
for i in range(0,3):
    minI.append(mape.index(min(mape)))
    minM.append(min(mape))
    mape[minI[i]] = 100

print("Neural Networks with minimum Mean absolute percentage error:")

for i in range(0,3):
    value = minM[i]*100
    print("neural network : Hidden layer1:",Neural_Nets[minI[i]].hidden_layer_sizes[0],"Hidden layer2 :",Neural_Nets[minI[i]].hidden_layer_sizes[1])
    print("Mean absolute percentage error :",value,"\n")








