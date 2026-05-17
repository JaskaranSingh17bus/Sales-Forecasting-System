# SALES FORECASTING
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

# LOAD DATA
df = pd.read_csv('train.csv')

# E.D.A
# print(df.info())
# print(df.describe())
# print("Empty Spaces in Data: ", df.isnull().sum())

# CLEANING
df['Postal Code'] = df['Postal Code'].fillna(df['Postal Code'].mode()) 
df.dropna(subset=['Postal Code'], inplace=True)
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Month_Year'] = df['Order Date'].dt.to_period('M').astype(str)
daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()
monthly_sales = df.groupby('Month_Year')['Sales'].sum().reset_index()
top_cities = df.groupby('City')['Sales'].sum().nlargest(10).reset_index()
#PRE-PROCESSING
# print(df.isnull().sum())

# VISUALING DATA    

sns.set_style('whitegrid')

plt.figure(figsize=(12,8))
plt.title('Time Series Graph')
sns.lineplot(x='Month_Year', y='Sales', data=monthly_sales, marker='o', color='purple')
plt.xticks(rotation=45)
plt.title('Time Series Graph')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(12,8))
sns.barplot(x='Category', y='Sales', data=df, estimator=sum, palette='viridis')
plt.title('Category-wise Analysis')
# plt.ytitle('Total Sales')
plt.tight_layout
plt.show()
plt.close

plt.figure(figsize=(12,8))
sns.barplot(x='Sales', y='City', data=top_cities, palette='mako')
plt.title('Sales in State/Region/City')
plt.xlabel('Total Sales')
plt.tight_layout()
plt.show()
plt.close()

# PRE-PROCESSING DATA
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day

from sklearn.model_selection import train_test_split
# CONVERTIN DATA IN 0's & 1's
df_encoded = pd.get_dummies(df, columns=['Category', 'Region', 'Segment'])

X = df_encoded.select_dtypes(include=['number'])
X = X.drop(['Sales'], axis=1, errors='ignore')
y = df_encoded['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Total data: {len(X)}")
print(f"Training Data: {len(X_train)}")
print(f"Testing Data: {len(X_test)}")

# TRAINING MODEL
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# INTALIZING MODEL
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# FITTING DATA IN MODEL
model.fit(X_train, y_train)

# PREDICTING 
y_pred = model.predict(X_test)

# CHECKING ACCURACY
score = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Model R2 Score: {score}")
print(f"Mean Absolute Error: {mae}")