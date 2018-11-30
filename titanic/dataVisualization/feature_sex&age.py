import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Read data
train_data = pd.read_csv("../train.csv")
test_data = pd.read_csv("../test.csv")

# Feature:Sex and Age
# 男性中老年人多，女性更年轻，小孩中男孩较多
f, ax = plt.subplots(figsize=(8,3))
ax.set_title('Sex Age dist', size=20)
sns.distplot(train_data[train_data.Sex == 'female'].dropna().Age, hist=False, color='pink', label='female')
sns.distplot(train_data[train_data.Sex == 'male'].dropna().Age, hist=False, color='blue', label='male')
ax.legend(fontsize=15)

plt.show()