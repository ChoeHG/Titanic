import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Read data
train_data = pd.read_csv("../train.csv")
test_data = pd.read_csv("../test.csv")

# Feature:Embarked
plt.style.use('ggplot')
ax = plt.figure(figsize=(8, 3)).add_subplot(111)
pos = [1, 2, 3]
y1 = train_data[train_data.Survived == 0].groupby('Embarked')['Survived'].count().sort_index().values
y2 = train_data[train_data.Survived == 1].groupby('Embarked')['Survived'].count().sort_index().values
ax.bar(pos, y1, color='r', alpha=0.4, align='center', label='dead')
ax.bar(pos, y2, color='g', alpha=0.4, align='center', label='alive', bottom=y1)
ax.set_xticks(pos)
ax.set_xticklabels(['C', 'Q', 'S'])
ax.legend(fontsize=15, loc='best')
ax.set_title('Embarked survived count', size=18)

plt.show()

ax = plt.figure(figsize=(8,3)).add_subplot(111)
ax.set_xlim([-20, 80])
sns.kdeplot(train_data[train_data.Embarked=='C'].Age.fillna(-10), ax=ax, label='C')
sns.kdeplot(train_data[train_data.Embarked=='Q'].Age.fillna(-10), ax=ax, label='Q')
sns.kdeplot(train_data[train_data.Embarked=='S'].Age.fillna(-10), ax=ax, label='S')
ax.legend(fontsize=18)
ax.set_title('Embarked Age Dist ', size=18)

plt.show()

y1 = train_data[train_data.Survived==0].groupby(['Embarked','Pclass'])['Survived'].count().reset_index()['Survived'].values
y2 = train_data[train_data.Survived==1].groupby(['Embarked','Pclass'])['Survived'].count().reset_index()['Survived'].values

ax = plt.figure(figsize=(8,3)).add_subplot(111)
pos = range(9)
ax.bar(pos, y1, align='center', alpha=0.5, color='r', label='dead')
ax.bar(pos, y2, align='center', bottom=y1, alpha=0.5, color='g', label='alive')

ax.set_xticks(pos)
xticklabels = []
for embarked_val in ['C','Q','S']:
    for pclass_val in range(1,4):
        xticklabels.append('%s/%d'%(embarked_val,pclass_val))

ax.set_xticklabels(xticklabels,size=15)
ax.legend(fontsize=15, loc='best')

plt.show()