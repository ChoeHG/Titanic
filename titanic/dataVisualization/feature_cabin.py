import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Read data
train_data = pd.read_csv("../train.csv")
test_data = pd.read_csv("../test.csv")

# Feature:Cabin
# Cabin 大部分为空，为空的获救概率较低，不为空的获救概率较高
print("^"*50)
print(train_data.Cabin.isnull().value_counts())
print(train_data.groupby(by=train_data.Cabin.isnull())['Survived'].mean())

# 把每个Cabin中的区域提取出来，不同区域获救的概率差别很大
train_data['Cabin_Zone'] = train_data.Cabin.fillna('0').str.split(' ').apply(lambda x: x[0][0])
print(train_data.groupby(by='Cabin_Zone')['Survived'].agg(['mean', 'count']))