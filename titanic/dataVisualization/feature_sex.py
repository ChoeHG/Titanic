import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Read data
train_data = pd.read_csv("../train.csv")
test_data = pd.read_csv("../test.csv")

# Feature:Sex
print(train_data.Sex.value_counts())
print('********************************')
print(train_data.groupby('Sex')['Survived'].mean())

ax = plt.figure(figsize=(10, 4)).add_subplot(111)
sns.violinplot(x='Sex', y='Age', hue='Survived', data=train_data.dropna(), split=True)
ax.set_xlabel('Sex', size=20)
ax.set_xticklabels(['Female', 'male'], size=18)
ax.set_ylabel('Age', size=20)
ax.legend(fontsize=25, loc='best')

plt.show()