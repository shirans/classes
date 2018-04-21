import numpy as np




# normalize numerical columns
# one-hot categorical columns

df = pd.DataFrame([1,2,3,4,5,6,7,8])
df.describe()
std = df.std()
df2 = (df - df.mean())/ std

