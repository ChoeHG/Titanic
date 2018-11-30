import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Read data
train_data = pd.read_csv("../train.csv")
test_data = pd.read_csv("../test.csv")

# Feature:Sex and PClass
label = []
for sex_i in ['female','male']:
    for pclass_i in range(1,4):
        label.append('sex:%s,Pclass:%d'%(sex_i, pclass_i))

pos = range(6)
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(111)
ax.bar(pos,
        train_data[train_data['Survived'] == 0].groupby(['Sex', 'Pclass'])['Survived'].count().values,
        color='r',
        alpha=0.5,
        align='center',
        tick_label=label,
        label='dead')
ax.bar(pos,
        train_data[train_data['Survived'] == 1].groupby(['Sex', 'Pclass'])['Survived'].count().values,
        bottom=train_data[train_data['Survived'] == 0].groupby(['Sex', 'Pclass'])['Survived'].count().values,
        color='g',
        alpha=0.5,
        align='center',
        tick_label=label,
        label='alive')
ax.tick_params(labelsize=15)
ax.set_title('sex_pclass_survived', size=30)
ax.legend(fontsize=15, loc='best')

plt.show()