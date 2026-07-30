import pandas as pd
pd.set_option('display.max_columns',None)#打印时显示所有列
#从csv读取数据
df=pd.read_csv(r"C:\Users\Monki\Desktop\deep-learing\titanic\train.csv")
#去除不需要的列
df=df.drop(columns=["PassengerId","Name","Ticket","Cabin"])
#去除不需要的列
df=df.dropna(subset=["Age"])
#对Sex和Embarked做独热编码
df=pd.get_dummies(df,columns=["Sex","Embarked"],dtype=int)
#Dataset:负责数据读取和预处理

#Dataloader:将数据分成小批量，支持多线程加速以及数据的打乱
