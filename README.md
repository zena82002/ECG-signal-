# PTB-XL ECG Classifier

This project involves creating a classifier for the PTB-XL dataset to determine whether an ECG record is normal or abnormal.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The PTB-XL dataset is a large collection of electrocardiography (ECG) records. This project aims to build a classifier to identify whether a given ECG record is normal or abnormal using a Convolutional Recurrent Neural Network (CRNN) model.

## Dataset
The PTB-XL dataset is available at [PhysioNet](https://physionet.org/content/ptb-xl/1.0.1/). It contains 21,837 clinical 12-lead ECG records from 18,885 patients, each lasting 10 seconds and annotated with up to 71 different ECG statements.

## Model Architecture
The classifier is built using a CRNN architecture. The model consists of:
- Conv1D and MaxPooling1D layers for feature extraction.
- LSTM layers for capturing temporal dependencies.
- Dense layers for classification.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo.git
    ```
2. Navigate to the project directory:
    ```bash
    cd your-repo
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Prepare the dataset by downloading it from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.1/) and placing it in the appropriate directory.
2. Run the provided script to process the dataset and balance the classes:
    ```bash
    python process_dataset.py
    ```
3. Train the model:
    ```bash
    python train.py
    ```
4. Evaluate the model:
    ```bash
    python evaluate.py
    ```
5. Predict using the model:
    ```bash
    python predict.py --input path_to_ecg_record
    ```

## Results
The model achieves an accuracy of X% on the test set. Detailed performance metrics can be found in the `results` directory.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any changes or suggestions.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Detailed Code Explanation

### Data Preprocessing

# libraries

```python
import pandas as pd
import numpy as np
import wfdb
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
```

# Load and preprocess the dataset
```python
def preprocess_data():
    file_path = r"ptbxl_database.csv"
    data = pd.read_csv(file_path)
    norm_data = data[data['scp_codes'].str.contains('NORM')]
    imi_data = data[data['scp_codes'].str.contains('IMI')]
    min_records = min(norm_data.shape[0], imi_data.shape[0])
    norm_data = norm_data.sample(n=min_records, random_state=42)
    imi_data = imi_data.sample(n=min_records, random_state=42)
    final_data = pd.concat([norm_data, imi_data])
    final_data.to_csv('New_data.csv', index=False)
    return final_data

def collect_and_label(dataset):
    df = pd.read_csv(dataset)
    alpha = df['scp_codes'].str.split("'").str[1].str[-2:] == 'MI'
    beta = df['scp_codes'].str.split("'").str[1] == 'NORM'
    df = df[alpha | beta]
    df['label'] = df['scp_codes'].str.split("'").str[1]
    return df
```

# Apply high-pass filter
```python
def apply_highpass_filter(signal, lowcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='high')
    return filtfilt(b, a, signal)

def apply_highpass_filter_to_lead1(file_path):
    record = wfdb.rdrecord(file_path)
    lead_1_signal = record.p_signal[:, 0]
    fs = record.fs
    lowcut = 0.5
    filtered_lead_1_signal = apply_highpass_filter(lead_1_signal, lowcut, fs)
    return filtered_lead_1_signal
```

# Balance and augment the dataset
```python
def balance_and_augment(df):
    smote_nc = SMOTENC(categorical_features=[1], random_state=0)
    X_res, y_res = smote_nc.fit_resample(df[['ecg_id', 'filename_hr']].to_numpy(), df['label'])
    df_balanced = pd.DataFrame(X_res, columns=['ecg_id', 'filename_hr'])
    df_balanced['label'] = y_res
    return df_balanced
```

# Preprocess the signals
```python
def preprocess_signals(dat):
    data_dir = list(dat['filename_hr'])
    data_signal = list(map(read_signal, data_dir))
    data_signal = np.array(data_signal)
    data_dict = {'NORM': 0, 'IMI': 1}
    encoded_label = dat['label'].map(data_dict)
    return np.array(data_signal), np.array(encoded_label)
```

# Read the signal from the record
```python
def read_signal(record):
    tes = sig.rdrecord(record, sampfrom=0, sampto=5000)
    signal = tes.__dict__['p_signal'][:, 0]
    return signal
```

# Main preprocessing function
```python
def main():
    df_labeled = collect_and_label('New_data.csv')
    df_balanced_and_augmented = balance_and_augment(df_labeled)
    df_balanced_and_augmented = df_balanced_and_augmented.sample(frac=1, ignore_index=True, random_state=123)
    file_paths = df_balanced_and_augmented['filename_hr']
    filtered_signals = [apply_highpass_filter_to_lead1(file_path) for file_path in file_paths]
    df_balanced_and_augmented['filtered_lead_1_channel_0'] = filtered_signals

    train, val_test = train_test_split(df_balanced_and_augmented, train_size=0.7, random_state=1002)
    validation, test = train_test_split(val_test, test_size=0.5, random_state=1002)
    X_train, y_train = preprocess_signals(train)
    X_valid, y_valid = preprocess_signals(validation)
    X_test, y_test = preprocess_signals(test)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

if __name__ == "__main__":
    main()
```

## Model Training and Evaluation
# libraries 
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint
```
# Function to build the CRNN model
```python
def build_crnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model
```

# Train and evaluate the model
```python
def train_and_evaluate(X_train, y_train, X_valid, y_valid, X_test, y_test):
    input_shape = X_train.shape[1:]
    num_classes = 2
    crnn_model = build_crnn_model(input_shape, num_classes)
    crnn_model.compile(optimizer=Adam(learning_rate=0.0001), loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])
    crnn_model.summary()
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_sparse_categorical_accuracy', save_best_only=True, mode='max', verbose=1)
    history = crnn_model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_valid, y_valid), callbacks=[checkpoint])
    test_loss, test_accuracy =Here's a comprehensive README file for your project:
```

# PTB-XL ECG Classification

## Project Overview

This project aims to create a classifier for the PTB-XL dataset to determine whether an ECG record is normal (NORM) or abnormal (IMI). The classifier is built using a Convolutional Recurrent Neural Network (CRNN) model implemented in TensorFlow/Keras.

## Dataset

The dataset used is the PTB-XL, a large publicly available dataset of electrocardiograms (ECGs). It contains various annotated ECG records, including normal and abnormal classes.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- wfdb
- scikit-learn
- imbalanced-learn (imblearn)
- TensorFlow
- Keras

Install the required libraries using pip:

```bash
pip install pandas numpy matplotlib wfdb scikit-learn imbalanced-learn tensorflow keras
