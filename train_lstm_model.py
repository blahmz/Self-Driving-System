"""
BiLSTM Model Trainer
Trains on data collected in Teacher mode.
Sensor layout: 5 front + 6 diagonal + 5 rear = 16 sensors
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
import pickle

from config import (
    SEQUENCE_LENGTH, LSTM_UNITS_LAYER1, LSTM_UNITS_LAYER2,
    DROPOUT_RATE, DENSE_UNITS, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT,
    MIN_ACCURACY_THRESHOLD, TRAINING_DATA_FILE, MODEL_FILE,
    LABEL_ENCODER_FILE, SCALER_FILE,
)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow not found. Training unavailable.")


class LSTMModelTrainer:
    def __init__(self):
        self.model   = None
        self.encoder = None
        self.scaler  = None
        self.history = None

    def load_and_preprocess(self):
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        try:
            df = pd.read_csv(TRAINING_DATA_FILE)
            print(f"Loaded {len(df)} rows from {TRAINING_DATA_FILE}")
        except FileNotFoundError:
            print(f"ERROR: {TRAINING_DATA_FILE} not found.")
            print("Run the simulation in TEACHER mode first, then press ESC to save.")
            return None, None, None, None

        df = df[df['decision'] != 'decision']
        sensor_cols = [c for c in df.columns if 'sensor' in c]
        df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        print(f"Cleaned: {len(df)} rows, {len(sensor_cols)} sensor features")
        print("\nClass distribution:")
        print(df['decision'].value_counts())
        print()

        X_seqs, y_seqs = self._create_sequences(df, sensor_cols)
        if len(X_seqs) == 0:
            print("ERROR: Not enough data for sequences.")
            return None, None, None, None
        print(f"Created {len(X_seqs)} sequences of length {SEQUENCE_LENGTH}")

        self.encoder = LabelEncoder()
        y_enc  = self.encoder.fit_transform(y_seqs)
        y_cat  = tf.keras.utils.to_categorical(y_enc)

        self.scaler = StandardScaler()
        flat   = X_seqs.reshape(-1, X_seqs.shape[-1])
        flat   = self.scaler.fit_transform(flat)
        X_seqs = flat.reshape(X_seqs.shape)

        cw = class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_enc), y=y_enc)
        cw_dict = dict(enumerate(cw))
        print(f"Class weights: {cw_dict}")

        return X_seqs, y_cat, cw_dict, y_enc

    def _create_sequences(self, df, sensor_cols):
        data   = df[sensor_cols].values
        labels = df['decision'].values
        X, y   = [], []
        for i in range(len(data) - SEQUENCE_LENGTH):
            X.append(data[i:i + SEQUENCE_LENGTH])
            y.append(labels[i + SEQUENCE_LENGTH])
        return np.array(X), np.array(y)

    def build_model(self, input_shape, num_classes):
        print("\nBUILDING BiLSTM MODEL")
        print("=" * 60)
        self.model = Sequential([
            Bidirectional(LSTM(LSTM_UNITS_LAYER1, return_sequences=True),
                          input_shape=input_shape, name='bilstm_1'),
            Dropout(DROPOUT_RATE),
            Bidirectional(LSTM(LSTM_UNITS_LAYER2), name='bilstm_2'),
            Dropout(DROPOUT_RATE),
            Dense(DENSE_UNITS, activation='relu'),
            Dropout(DROPOUT_RATE / 2),
            Dense(num_classes, activation='softmax'),
        ])
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()
        print(f"Parameters: {self.model.count_params():,}")

    def train(self, X_tr, X_te, y_tr, y_te, cw):
        print("\nTRAINING")
        print("=" * 60)
        cb = [
            EarlyStopping(monitor='val_accuracy', patience=10,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=1e-6, verbose=1),
        ]
        self.history = self.model.fit(
            X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_data=(X_te, y_te), class_weight=cw,
            callbacks=cb, verbose=1)
        final_val = self.history.history['val_accuracy'][-1]
        print(f"\nFinal val accuracy: {final_val * 100:.2f}%")
        return final_val >= MIN_ACCURACY_THRESHOLD

    def save(self):
        self.model.save(MODEL_FILE)
        with open(LABEL_ENCODER_FILE, 'wb') as f:
            pickle.dump(self.encoder, f)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"\nSaved: {MODEL_FILE}, {LABEL_ENCODER_FILE}, {SCALER_FILE}")


def main():
    if not TF_AVAILABLE:
        print("TensorFlow required for training.")
        return

    trainer = LSTMModelTrainer()
    X, y_cat, cw, y_enc = trainer.load_and_preprocess()
    if X is None:
        return

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_enc)

    trainer.build_model(input_shape=(X.shape[1], X.shape[2]),
                        num_classes=y_cat.shape[1])
    success = trainer.train(X_tr, X_te, y_tr, y_te, cw)
    trainer.save()

    if success:
        print("\nTraining successful! Run simulation in AI mode (press A).")
    else:
        print("\nModel saved but accuracy below target. Collect more data.")


if __name__ == "__main__":
    main()
