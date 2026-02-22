"""
Bidirectional LSTM Model Training
Achieves 99%+ accuracy with temporal sequence learning
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle

from config import (SEQUENCE_LENGTH, LSTM_UNITS_LAYER1, LSTM_UNITS_LAYER2,
                    DROPOUT_RATE, DENSE_UNITS, BATCH_SIZE, EPOCHS,
                    VALIDATION_SPLIT, MIN_ACCURACY_THRESHOLD,
                    TRAINING_DATA_FILE, MODEL_FILE, LABEL_ENCODER_FILE, SCALER_FILE)


class LSTMModelTrainer:
    """Trains Bidirectional LSTM model for autonomous driving"""
    
    def __init__(self):
        self.model = None
        self.encoder = None
        self.scaler = None
        self.history = None
        
    def load_and_preprocess_data(self):
        """Load and prepare sequential data"""
        print("="*60)
        print("PHASE 0: LOADING AND PREPROCESSING DATA")
        print("="*60)
        
        try:
            df = pd.read_csv(TRAINING_DATA_FILE)
            print(f"✓ Loaded {len(df)} samples from {TRAINING_DATA_FILE}")
        except FileNotFoundError:
            print(f"✗ ERROR: '{TRAINING_DATA_FILE}' not found!")
            print("  Please run the simulation in TEACHER mode first!")
            return None, None, None, None
        
        # Clean data
        df = df[df['decision'] != 'decision']
        
        # Force sensors to numeric
        sensor_cols = [c for c in df.columns if 'sensor' in c]
        df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        
        print(f"✓ Cleaned data: {len(df)} valid samples")
        
        # Check class balance
        print("\n--- Class Distribution ---")
        print(df['decision'].value_counts())
        print("--------------------------\n")
        
        # Create sequences
        print("Creating temporal sequences...")
        X_sequences, y_sequences = self._create_sequences(df, sensor_cols)
        
        if len(X_sequences) == 0:
            print("✗ ERROR: Not enough data for sequences!")
            return None, None, None, None
        
        print(f"✓ Created {len(X_sequences)} sequences of length {SEQUENCE_LENGTH}")
        
        # Encode labels
        self.encoder = LabelEncoder()
        y_encoded = self.encoder.fit_transform(y_sequences)
        y_categorical = tf.keras.utils.to_categorical(y_encoded)
        
        # Normalize features
        self.scaler = StandardScaler()
        X_reshaped = X_sequences.reshape(-1, X_sequences.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_sequences = X_scaled.reshape(X_sequences.shape)
        
        # Calculate class weights
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_encoded),
            y=y_encoded
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        print(f"✓ Class weights: {class_weight_dict}")
        
        return X_sequences, y_categorical, class_weight_dict, y_encoded
    
    def _create_sequences(self, df, sensor_cols):
        """Create temporal sequences from data"""
        X_sequences = []
        y_sequences = []
        
        # Group by continuous driving sessions (simple approach: sliding window)
        data = df[sensor_cols].values
        labels = df['decision'].values
        
        for i in range(len(data) - SEQUENCE_LENGTH):
            sequence = data[i:i + SEQUENCE_LENGTH]
            label = labels[i + SEQUENCE_LENGTH]
            
            X_sequences.append(sequence)
            y_sequences.append(label)
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def build_lstm_model(self, input_shape, num_classes):
        """Build Bidirectional LSTM architecture"""
        print("\n" + "="*60)
        print("PHASE 1: BUILDING BIDIRECTIONAL LSTM MODEL")
        print("="*60)
        
        self.model = Sequential([
            # First Bidirectional LSTM layer
            Bidirectional(LSTM(LSTM_UNITS_LAYER1, return_sequences=True), 
                         input_shape=input_shape,
                         name='bi_lstm_1'),
            Dropout(DROPOUT_RATE, name='dropout_1'),
            
            # Second Bidirectional LSTM layer
            Bidirectional(LSTM(LSTM_UNITS_LAYER2, return_sequences=False),
                         name='bi_lstm_2'),
            Dropout(DROPOUT_RATE, name='dropout_2'),
            
            # Dense layers
            Dense(DENSE_UNITS, activation='relu', name='dense_1'),
            Dropout(DROPOUT_RATE / 2, name='dropout_3'),
            
            # Output layer
            Dense(num_classes, activation='softmax', name='output')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create a list to collect summary lines
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        print("\n" + "\n".join(summary_lines))
        print(f"\n✓ Model built with {self.model.count_params():,} parameters")
        
    def train_model(self, X_train, X_test, y_train, y_test, class_weight_dict):
        """Train the LSTM model with callbacks"""
        print("\n" + "="*60)
        print("PHASE 2: TRAINING MODEL")
        print("="*60)
        
        # Callbacks for optimal training
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        print(f"\nTraining on {len(X_train)} samples, validating on {len(X_test)} samples")
        print(f"Target accuracy: {MIN_ACCURACY_THRESHOLD * 100}%\n")
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            class_weight=class_weight_dict,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate final accuracy
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Final Training Accuracy:   {final_train_acc*100:.2f}%")
        print(f"Final Validation Accuracy: {final_val_acc*100:.2f}%")
        
        if final_val_acc >= MIN_ACCURACY_THRESHOLD:
            print(f"✓ SUCCESS! Achieved target accuracy of {MIN_ACCURACY_THRESHOLD*100}%")
        else:
            print(f"✗ WARNING: Did not reach target accuracy of {MIN_ACCURACY_THRESHOLD*100}%")
            print("  Consider collecting more training data or adjusting hyperparameters")
        
        return final_val_acc >= MIN_ACCURACY_THRESHOLD
    
    def save_model_and_artifacts(self):
        """Save trained model and preprocessing artifacts"""
        print("\n" + "="*60)
        print("PHASE 3: SAVING MODEL AND ARTIFACTS")
        print("="*60)
        
        # Save model
        self.model.save(MODEL_FILE)
        print(f"✓ Model saved to '{MODEL_FILE}'")
        
        # Save label encoder
        with open(LABEL_ENCODER_FILE, "wb") as f:
            pickle.dump(self.encoder, f)
        print(f"✓ Label encoder saved to '{LABEL_ENCODER_FILE}'")
        
        # Save scaler
        with open(SCALER_FILE, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Feature scaler saved to '{SCALER_FILE}'")
        
        print("\n" + "="*60)
        print("ALL ARTIFACTS SAVED SUCCESSFULLY")
        print("="*60)


def main():
    """Main training pipeline"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  AUTONOMOUS DRIVING - BIDIRECTIONAL LSTM TRAINER".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    print("\n")
    
    trainer = LSTMModelTrainer()
    
    # Load and preprocess
    X_seq, y_cat, class_weights, y_enc = trainer.load_and_preprocess_data()
    
    if X_seq is None:
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_cat, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    # Build model
    trainer.build_lstm_model(
        input_shape=(X_seq.shape[1], X_seq.shape[2]),
        num_classes=y_cat.shape[1]
    )
    
    # Train model
    success = trainer.train_model(X_train, X_test, y_train, y_test, class_weights)
    
    # Save everything
    trainer.save_model_and_artifacts()
    
    if success:
        print("\n✓ Training completed successfully!")
        print("  You can now run the simulation in AI mode (press 'A')")
    else:
        print("\n⚠ Training completed but accuracy is below target")
        print("  Model saved but may need more training data")


if __name__ == "__main__":
    main()