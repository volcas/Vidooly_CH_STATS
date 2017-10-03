# Vidooly_CH_STATS
Stats Prediction made for the next 3 months of YouTube channel with channel id = 'UC6f5t2CAXGY6-ju7Ug' based on last 6 months statistics i.e. Views, Subscriber count, Video count

New to RNN,

Code inspired by https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/


Views is predicted with RMSE=3925.009

Subscriber count is predicted with RMSE=2.906

Video count is predicted with RMSE=0.013

Note: RMSE can be reduced by change in no. of epochs and neurons

File Description:
  
 Combined.py -Contains code to combine the three CSV prediction files to one

 Dataset.csv -Contains the data which was used for prediction
 
 FinalOP.csv -Contains prediction of Views, Subscriber Count, Video Count from 2017-04-01 to 2017-06-30
 
 SubscriberCount.csv -Contains predicted SubcriberCount 		
 
 SubscriberCountPrediction.py -Contains code to predict Subscriber count
 
 VideoCount.csv  -Contains predicted Video Count
 
 VidCountPrediction.py 	-Contains code to predict Video Count
 
 Views.csv -Contains predicted Views
 
 ViewsPrediction.py -Contains code to predict Views 
