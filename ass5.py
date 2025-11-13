import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------
# Step 1: Load Dataset
# -------------------------
df = pd.read_csv("SCOA_A5.csv")

# Rename columns if needed
df = df.rename(columns={
    'date': 'Date', 'open': 'Open', 'high': 'High',
    'low': 'Low', 'close': 'Close', 'volume': 'Volume'
})

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# -------------------------
# Step 2: Create Target
# -------------------------
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Features & labels
features = df[['Open', 'High', 'Low', 'Close', 'Volume']]
X = MinMaxScaler().fit_transform(features)
y = df['Target'].values

# Remove last row due to shift
X, y = X[:-1], y[:-1]

# -------------------------
# Step 3: Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -------------------------
# Step 4: ANN Model
# -------------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(5,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------
# Step 5: Train Model
# -------------------------
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# -------------------------
# Step 6: Evaluate
# -------------------------
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
