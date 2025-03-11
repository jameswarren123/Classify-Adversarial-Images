import pandas as pd

df0 = pd.read_csv('refactor\mnist_train.csv')
df1 = pd.read_csv('refactor\mnist_test.csv')
columns = ['number_label'] + [f'pixel_{i}' for i in range(784)]
df0.columns = columns
df1.columns = columns
#df0.to_pickle('mnist_train.pkl')
#df1.to_pickle('mnist_test.pkl')
print(df0)
print(df1)