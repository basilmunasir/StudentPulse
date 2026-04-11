import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/student_habits_performance.csv')

# Fix missing values
df['parental_education_level'] = df['parental_education_level'].fillna(df['parental_education_level'].mode()[0])

# Select features based on heatmap
features = [
    'study_hours_per_day',
    'mental_health_rating',
    'exercise_frequency',
    'sleep_hours',
    'social_media_hours',
    'netflix_hours'
]

X = df[features]
y = df['exam_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build TensorFlow model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)

# Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f"\nModel MAE: {mae:.2f}")
print(f"This means predictions are off by {mae:.2f} marks on average")

# Save model
model.save('models/studentpulse_model.keras')
print("\nModel saved successfully!")

# Plot training history
plt.figure(figsize=(8,5))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Training Progress')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()