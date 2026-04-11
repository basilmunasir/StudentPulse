import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data/student_habits_performance.csv')

# Basic exploration
print("Shape: ", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic Stats:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values in parental_education_level with the most common value
most_common = df['parental_education_level'].mode()[0]
df['parental_education_level'] = df['parental_education_level'].fillna(most_common)
# Verify it's fixed
print("Missing values after fix:")
print(df.isnull().sum())

# Set style
sns.set_style("darkgrid")

# 1. Distribution of Exam Scores
plt.figure(figsize=(8,5))
sns.histplot(df['exam_score'], bins=30, kde=True, color='blue')
plt.title('Distribution of Exam Scores')
plt.xlabel('Exam Score')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.savefig('exam_score_distribution.png')
plt.show()

# 2. Sleep Hours vs Exam Score
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['sleep_hours'], y=df['exam_score'], color='green')
plt.title('Sleep Hours vs Exam Score')
plt.xlabel('Sleep Hours')
plt.ylabel('Exam Score')
plt.tight_layout()
plt.savefig('sleep_vs_score.png')
plt.show()

# 3. Study Hours vs Exam Score
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['study_hours_per_day'], y=df['exam_score'], color='orange')
plt.title('Study Hours vs Exam Score')
plt.xlabel('Study Hours Per Day')
plt.ylabel('Exam Score')
plt.tight_layout()
plt.savefig('study_vs_score.png')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include='number').corr(),annot=True,fmt='.2f',cmap='coolwarm',
            linewidths=0.5)
plt.title('Correlation Heatmap - What affects Exam Scores?')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

