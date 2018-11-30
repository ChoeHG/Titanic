import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Read data
train_data = pd.read_csv("../train.csv")
test_data = pd.read_csv("../test.csv")

# Feature:Fare
# 出钱多的人，更容易获救
fig = plt.figure(figsize=(8, 6))
ax = plt.subplot2grid((2,2), (0,0), colspan=2)

ax.tick_params(labelsize=15)
ax.set_title('Fare dist', size=20)
ax.set_ylabel('dist', size=20)
sns.kdeplot(train_data.Fare, ax=ax)
sns.distplot(train_data.Fare, ax=ax)
ax.legend(fontsize=15)
pos = range(0,400,50)
ax.set_xticks(pos)
ax.set_xlim([0, 200])
ax.set_xlabel('')

ax1 = plt.subplot2grid((2,2), (1,0), colspan=2)
ax.set_title('Fare Pclass dist', size=20)
for i in range(1, 4):
    sns.kdeplot(train_data[train_data.Pclass == i].Fare, ax=ax1, label='Pclass %d' % i)
ax1.set_xlim([0, 200])
ax1.legend(fontsize=15)

plt.show()

fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(111)
sns.kdeplot(train_data[train_data.Survived == 0].Fare, ax=ax1, label='dead', color='r')
sns.kdeplot(train_data[train_data.Survived == 1].Fare, ax=ax1, label='alive', color='g')
ax1.set_xlim([0, 300])
ax1.legend(fontsize=15)
ax1.set_title('Fare survived', size=20)
ax1.set_xlabel('Fare', size=15)

plt.show()