import pandas as pd
import matplotlib.pyplot as plt

FILENAME = 'data.csv'
df = pd.read_csv(FILENAME)

print('### BASIC STATISTICAL DESCRIPTION ###')
print(df.describe())
print()

print('### NULL CALORIE VALUES BEFORE REMOVAL ###')
print(df[df.isnull()['Calories']])

calMean = df['Calories'].mean()
df['Calories'] = df['Calories'].fillna(calMean)

print('### NULL CALORIE VALUES AFTER REMOVAL ###')
print(df[df.isnull()['Calories']])
print()

print('### AGGREGATED DATA ###')
print(df.agg({ 'Pulse': ['min', 'max', 'count', 'mean'], 'Maxpulse': ['min', 'max', 'count', 'mean'] }))
print()

print('### RECORDS WITH CALORIE VALUES BETWEEN 500 AND 1000 ###')
print(df[(df['Calories'] > 500) & (df['Calories'] < 1000)])
print()

print('### RECORDS WITH CALORIE VALUES ABOVE 500 AND PULSES UNDER 100 ###')
print(df[(df['Calories'] > 500) & (df['Pulse'] < 100)])
print()

print('### df_modified ###')
df_modified = df.filter(['Duration', 'Pulse', 'Calories'])
print(df_modified)
print()

print('### df WITH COLUMN "Maxpulse" DELETED ###')
df = df_modified
print(df)
print()

print('### CONVERT CALORIES TO INT ###')
df['Calories'] = df['Calories'].astype(int)
print(df)
print()

df.plot.scatter('Duration', 'Calories')
plt.show()
