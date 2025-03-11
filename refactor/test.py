import pandas as pd

df0 = pd.read_csv('refactor\mnist_train.csv')
df1 = pd.read_csv('refactor\mnist_test.csv')

df0.to_pickle('mnist_train.pkl')
df1.to_pickle('mnist_test.pkl')