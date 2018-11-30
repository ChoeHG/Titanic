import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Read data
train_data = pd.read_csv("../train.csv")
test_data = pd.read_csv("../test.csv")

# Feature:PClass and Age
f, ax = plt.subplots(figsize=(8, 3))
ax.set_title('Pclass Age dist', size=20)
sns.distplot(train_data[train_data.Pclass == 1].dropna().Age, hist=False, color='pink', label='P1')
sns.distplot(train_data[train_data.Pclass == 2].dropna().Age, hist=False, color='blue', label='P2')
sns.distplot(train_data[train_data.Pclass == 3].dropna().Age, hist=False, color='g', label='P3')
ax.legend(fontsize=15)

plt.show()