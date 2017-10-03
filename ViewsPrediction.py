
#Importing necessary header files
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn import preprocessing
from pandas import DataFrame
from pandas import concat
from pandas import Series

#Fixing the random.seed so it produces the same result everytime
np.random.seed(7)

#Function to convert time series sequence to supervised 
def Supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df
 
#Function to create a differenced series
def differencedSeries(Data, length=1):
	Sub = list()
	for i in range(length, len(Data)):
		value = Data[i] - Data[i - length]
		Sub.append(value)
	return Series(Sub)
 
#Function to invert the differenced value
def Invert(Previous, Value):

   return Value + Previous
 
#Function to scale the test and train dataset 
def FuncScale(train, FinalInput):
	# fit scaler
	ScalerModel = preprocessing.MinMaxScaler(feature_range=(-1, 1))
	ScalerModel = ScalerModel.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = ScalerModel.transform(train)
	# transform FinalInput
	FinalInput = FinalInput.reshape(FinalInput.shape[0], FinalInput.shape[1])
	FinalInput_scaled = ScalerModel.transform(FinalInput)
	return ScalerModel, train_scaled, FinalInput_scaled
 
#Function to inverse scaling for a predicted value
def FuncInvertScale(ScalerModel, Data, value):
	next_row = [Data] + [value]
	array = np.array(next_row)
	array = array.reshape(1, len(array))
	Final = ScalerModel.inverse_transform(array)
	return Final[0, -1]
 
#Function to fit the LSTM network to training data
def FIT(train, batch_size, epoch, neuron):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neuron, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model
 
#Function to make a one-step prediction
def Predict(model, batch_size, Data):
	Data = Data.reshape(1, 1, 1)
	Output = model.predict(Data, batch_size=batch_size)
	return Output[0,0]
 
#Loading dataset
DataSet= pd.read_csv("Dataset.CSV",usecols=[1])

#Loading the last two values of the dataset for LSTM to use
FinalInput=[6959154,6964029]
 
#Transforming data to be stationary
Data_array = DataSet.values
Diff_array = differencedSeries(Data_array, 1)
FinalInput_Diff_array=differencedSeries(FinalInput, 1)
 
#Transforming data to be supervised 
SupervisedArray = Supervised(Diff_array, 1)
SupervisedArray_values = SupervisedArray.values
FinalInput_supervised = Supervised(FinalInput_Diff_array, 1)
FinalInput_supervised_values = FinalInput_supervised.values
 
#Renaming Data into train and FinalInput-sets
train=  SupervisedArray_values 

FinalInput= FinalInput_supervised_values 

#Transforming the scale of the data for better calculation
ScalerModel, train_scaled, FinalInput_scaled = FuncScale(train, FinalInput)
 
#Fitting the model
LSTMModel = FIT(train_scaled, 1, 1500, 1)
#Forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
LSTMModel.predict(train_reshaped, batch_size=1)

#The input for prediction
Last_Value= FinalInput_scaled[0, 0:-1]

#Walk-forward validation on the final input
New_predictions = pd.DataFrame({'Views' : []})

#To help with the difference
Prev_value=6964029

#Prediction
for i in range(90):
	
	X=Last_Value
	OP = Predict(LSTMModel, 1, X)
	Last_Value=OP# invert scaling
	OP = FuncInvertScale(ScalerModel, X, OP)
	# invert differencing
	OP = Invert(Prev_value,OP)
	
   # storing forecast
	New_predictions= New_predictions.append(pd.DataFrame({'Views':OP},index=[i+1]))
	
	Prev_value=OP    #Changing last value to the current value for the next iteration
    

print(New_predictions)
New_predictions.to_csv("Views.csv",index=False)
