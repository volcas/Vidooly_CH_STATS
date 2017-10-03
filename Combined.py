import pandas as pd

Views=pd.read_csv('Views.csv')
SubCount=pd.read_csv('SubscriberCount.csv')
VidCount=pd.read_csv('VideoCount.csv')



FinalOP= pd.concat([Views['Views'], SubCount['SubCount'], VidCount['VideoCount']], axis=1)
print(FinalOP)

FinalOP.to_csv('FinalOP.csv',index=False)

