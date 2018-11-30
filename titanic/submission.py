import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor

warnings.filterwarnings('ignore')

# Read data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
data = train_data.append(test_data, ignore_index=True)

# Feature Engineering
print('***********Train*************')
print('titanic')
print(train_data.isnull().sum())
print('***********titanic*************')
print(test_data.isnull().sum())


# 数据填充
# 使用 RandomForestClassifier 填补缺失的年龄属性
def fill_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regression中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    x = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1:])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df


data['Cabin'] = data['Cabin'].fillna('U')

data['Embarked'] = data['Embarked'].fillna('S')

data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

data = fill_missing_ages(data)


data.info()

# 数据处理：Sex转换为可操作的0和1,将Embarked转换为1,2,3

print(data["Sex"].unique())
data.loc[data["Sex"] == "male", "Sex"] = 1
data.loc[data["Sex"] == "female", "Sex"] = 0
print(data["Sex"].unique())

print(data["Embarked"].unique())
data["Embarked"] = data["Embarked"].fillna('S')
data.loc[data["Embarked"] == "S", "Embarked"] = 1
data.loc[data["Embarked"] == "C", "Embarked"] = 2
data.loc[data["Embarked"] == "Q", "Embarked"] = 3
print(data["Embarked"].unique())

# 数据处理：合并SibSp与Parch

data['Family'] = data['SibSp'] + data['Parch']

data_new = pd.concat([data['Age'],
                      data['Pclass'],
                      data['Family'],
                      data['Fare'],
                      data['Embarked'],
                      data['Sex'],
                      ], axis=1)

data_info = data_new.loc[0:890, :]
data_labels = data.loc[0:890, 'Survived', ]
pre_X = data_new.loc[891:, :]

train_X, test_X, train_y, test_y = train_test_split(data_info, data_labels, train_size=.8)
model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.03, max_depth=3)
model.fit(train_X, train_y)

print(model.score(test_X, test_y))


pre_Y = model.predict(pre_X)
pre_Y = pre_Y.astype(int)
passenger_id = data.loc[891:, 'PassengerId']
submission = pd.DataFrame(
    {'PassengerId': passenger_id,
     'Survived': pre_Y})

submission.to_csv('titanic_pred.csv', index=False)

