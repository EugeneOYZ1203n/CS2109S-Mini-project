import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from utils import generate_sklearn_loader_snippet


df = pd.read_csv('data/cipher_objective.csv')

new_columns_df = df['text'].str.split('', expand=True)
new_columns_df = new_columns_df.iloc[:, 1:-1] 
new_columns_df.columns = [f'char_{i+1}' for i in range(new_columns_df.shape[1])]
new_columns_df = new_columns_df.map(ord)
new_columns_df = (new_columns_df - 33) / (126 - 33) # 33 is min val, 126 is max val
df = pd.concat([df, new_columns_df], axis=1)

#print(new_columns_df.describe())

X = df.drop(columns=['class', 'text'])

y = df['class']   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000, random_state=42, C=1, solver='lbfgs')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nTest Set Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

loader_snippet = generate_sklearn_loader_snippet(model)

# Define output file path
output_path = "model_loader_snippet.txt"

# Write to file
with open(output_path, "w", encoding="utf-8") as f:
    f.write(loader_snippet)

print(f"✅ Loader snippet saved to {output_path}")