import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Read data
train_data = pd.read_csv("../train.csv")
test_data = pd.read_csv("../test.csv")

# Feature:Ticket
print("-" * 30)
print(train_data.Ticket.head())
print(train_data.Ticket.nunique())
print(train_data[train_data.Ticket == '110152'])