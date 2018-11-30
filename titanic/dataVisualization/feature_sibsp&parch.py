import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Read data
train_data = pd.read_csv("../train.csv")
test_data = pd.read_csv("../test.csv")

# sibsp & parch 表亲和直亲
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(211)
sns.countplot(train_data.SibSp)
ax1.set_title('SibSp', size=20)
ax2 = fig.add_subplot(212, sharex=ax1)
sns.countplot(train_data.Parch)
ax2.set_title('Parch', size=20)

plt.show()

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(311)
train_data.groupby('SibSp')['Survived'].mean().plot(kind='bar', ax=ax1)
ax1.set_title('Sibsp Survived Rate', size=16)
ax1.set_xlabel('')

ax2 = fig.add_subplot(312)
train_data.groupby('Parch')['Survived'].mean().plot(kind='bar', ax=ax2)
ax2.set_title('Parch Survived Rate', size=16)
ax2.set_xlabel('')

ax3 = fig.add_subplot(313)
train_data.groupby(train_data.SibSp+train_data.Parch)['Survived'].mean().plot(kind='bar', ax=ax3)
ax3.set_title('Parch+Sibsp Survived Rate', size=16)

plt.show()