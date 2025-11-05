from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import pandas as pd

from utils import generate_sklearn_loader_snippet

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CharToOrdTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_ = [f'char_{i+1}' for i in range(100)]
    
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X):
        transformed_rows = []
        for text in X:
            text = str(text)

            ascii_values = [ord(c) for c in text]
            transformed_rows.append(ascii_values)
        
        # return DataFrame with consistent column names
        return pd.DataFrame(transformed_rows, columns=self.columns_)

df = pd.read_csv('data/cipher_objective.csv')

# Basic info about the dataset
'''
print("=== Dataset Overview ===")
print(df.info())           # Columns, non-null counts, datatypes
print("\n=== First few rows ===")
print(df.head(5))          # Show first few rows

print("\n=== Class Distribution ===")
print(df['class'].value_counts())
'''

#df['length'] = df['text'].apply(len)
#print(df['length'].describe()) ## All exactly 100 long

#-------------------------------------------------Very no good metric: Split by char and convert to 
# Accuracy is around 50%

'''
unique_chars = set(''.join(df['text']))
print("Unique characters found:")
print(''.join(sorted(unique_chars)))
print(f"\nTotal unique characters: {len(unique_chars)}")

unique_chars_dict = {ch : i for i, ch in enumerate(unique_chars)}
char_to_int_fn = lambda c: unique_chars_dict[c]

new_columns_df = df['text'].str.split('', expand=True)
new_columns_df = new_columns_df.iloc[:, 1:-1] 
new_columns_df.columns = [f'char_{i+1}' for i in range(new_columns_df.shape[1])]
new_columns_df = new_columns_df.applymap(char_to_int_fn)
df = pd.concat([df, new_columns_df], axis=1)
'''



#-------------------------------------------------Frequency of each char
# Still 50%
# vectorizer = CountVectorizer(analyzer='char')

# -------------------------------------------TF-IDF weighting of each char
# vectorizer = TfidfVectorizer(analyzer='char')

#X = vectorizer.fit_transform(df['text'])

# ------------------------------------------- ASCII conversion

#new_columns_df = df['text'].str.split('', expand=True)
#new_columns_df = new_columns_df.iloc[:, 1:-1] 
#new_columns_df.columns = [f'char_{i+1}' for i in range(new_columns_df.shape[1])]
#new_columns_df = new_columns_df.applymap(ord)
#df = pd.concat([df, new_columns_df], axis=1)

#X = df.drop(columns=['class', 'text'])

X = df['text']

y = df['class']   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipe = Pipeline([
    ('char2ord', CharToOrdTransformer()),
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
])

param_grid = {
    'logreg__C': [0.1, 1, 10],
    'logreg__solver': ['lbfgs', 'liblinear']
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,                      # 5-fold CV
    n_jobs=-1,                 # use all CPU cores
    verbose=2,                 # show progress
    scoring='accuracy'         # or 'f1_macro', 'precision', etc.
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Display best parameters and model
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nTest Set Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

loader_snippet = generate_sklearn_loader_snippet(best_model)

# Define output file path
output_path = "model_loader_snippet.txt"

# Write to file
with open(output_path, "w", encoding="utf-8") as f:
    f.write(loader_snippet)

print(f"✅ Loader snippet saved to {output_path}")

print(best_model.predict(["O/#S$64gwuz8~-eW-h-GzRim>)(`v,7h^P<n-BX{<CV+M6T6w<}3i@*""0lS0gp5lCchQV>y.ILHyGD#{+T}_c}Ogs-::;vUgCOn@1"]))


'''
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
])

param_grid = {
    'svm__kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 'auto']
}

# Create grid search with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,                      # 5-fold CV
    n_jobs=-1,                 # use all CPU cores
    verbose=2,                 # show progress
    scoring='accuracy'         # or 'f1_macro', 'precision', etc.
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Display best parameters and model
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nTest Set Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
'''

'''
pipe = Pipeline([
    ('char2ord', CharToOrdTransformer()),
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='linear', C=10, gamma='scale'))
])

pipe = pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print("\nTest Set Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

loader_snippet = generate_sklearn_loader_snippet(pipe)

# Define output file path
output_path = "model_loader_snippet.txt"

# Write to file
with open(output_path, "w", encoding="utf-8") as f:
    f.write(loader_snippet)

print(f"✅ Loader snippet saved to {output_path}")

print(pipe.predict(["O/#S$64gwuz8~-eW-h-GzRim>)(`v,7h^P<n-BX{<CV+M6T6w<}3i@*""0lS0gp5lCchQV>y.ILHyGD#{+T}_c}Ogs-::;vUgCOn@1"]))
'''

'''
Best Parameters: {'svm__C': 10, 'svm__gamma': 'scale', 'svm__kernel': 'linear'}
Best Cross-Validation Accuracy: 0.96

Test Set Accuracy: 0.9783333333333334
Classification Report:
               precision    recall  f1-score   support

     default       0.99      0.97      0.98       622
        exit       0.97      0.99      0.98       578

    accuracy                           0.98      1200
   macro avg       0.98      0.98      0.98      1200
weighted avg       0.98      0.98      0.98      1200
'''

#scaler = StandardScaler()

#pca = PCA(n_components=0.95)
#svd = TruncatedSVD(n_components=40, random_state=42)

#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

#X_train_pca = pca.fit_transform(X_train_scaled)
#X_test_pca = pca.transform(X_test_scaled)

#### FINNDND DA ELBOOWW - WA DA HELL THERE IS NO ELBOW, maybe no need PCA
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('Number of Components')
#plt.ylabel('Cumulative Explained Variance')

#plt.show()


#param_grid = {
#    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
#    'C': [0.01, 0.1, 1, 10],
#    'gamma': ['scale', 'auto']
#}

#grid = GridSearchCV(SVC(random_state=42, shrinking=False), param_grid, cv=5, verbose=2, n_jobs=-1)
#grid.fit(X_train_pca, y_train)


#svm_classifier = SVC(kernel='linear', C=0.01, gamma="scale", random_state=42, shrinking=False, verbose=True)
#Linear -> 0.90
#Poly -> 0.89
#RBF -> 0.89
#Sigmoid -> 0.90


#print("Explained variance ratio (sum):", svd.explained_variance_ratio_.sum())

#print("Explained variance ratio:", pca.explained_variance_ratio_)
#print("Total variance retained:", sum(pca.explained_variance_ratio_))
#print("Num Components: ", len(pca.components_))

#svm_classifier.fit(X_train_pca, y_train)
#y_pred = svm_classifier.predict(X_test_pca)
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy:.2f}")