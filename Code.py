import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from keras_tuner import RandomSearch, HyperParameters
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Input
from keras.regularizers import l1_l2

# Enable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# Data Loading and Preprocessing
file_name = 'Dataset.csv'
file_path = os.path.join(data_dir, file_name)
df = pd.read_csv(file_path)
df['Y'] = df['Y'].replace(86, 0)

X = df.drop(['DrugID_A', 'DrugA', 'DrugID_B', 'DrugB', 'Y'], axis=1)
y = df['Y']
X_normalized = tf.keras.utils.normalize(X, axis=1).numpy()

# Splitting data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.33)

# Ensure TensorFlow compatibility
tf.compat.v1.reset_default_graph()

# Model Training and Evaluation Function
def train_evaluate_model(model, X, y):
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    scores = []
    for train_idx, test_idx in kfold.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        scores.append(accuracy_score(y[test_idx], preds))
    print("Model Accuracy: {:.2%}".format(np.mean(scores)))
    print("Classification Report:\n", classification_report(y[test_idx], preds))

# Deep Learning Model Setup
def build_dl_model(hp):
    model = keras.Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(units=hp.Int('units', min_value=64, max_value=256, step=32)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=hp.Float('dropout_rate', 0, 0.5, step=0.1)))
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(units=hp.Int(f'layer_{i}_units', 32, 128, step=32),
                        kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(rate=hp.Float(f'layer_{i}_dropout', 0.1, 0.5, step=0.1)))
    model.add(Dense(86, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameter Tuning Setup
tuner = RandomSearch(
    build_dl_model,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=3,
    directory='output_dir',
    project_name='DDI_HyperTuning'
)

# Early Stopping and Reduce Learning Rate on Plateau
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
]

# Search for best hyperparameters
tuner.search(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=callbacks)

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Bagging Ensemble
bagging_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)

# AdaBoost Ensemble
adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42)

# Stacking Ensemble
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10)),
    ('lr', LogisticRegression())
]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)

# Train and evaluate using K-Fold Cross-validation
train_evaluate_model(best_model, X_normalized, y)
train_evaluate_model(bagging_model, X_normalized, y)
train_evaluate_model(adaboost_model, X_normalized, y)
train_evaluate_model(stacking_model, X_normalized, y)
