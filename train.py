# train.py — FULLY AUTOMATIC: Finds & saves the TRUE best model
import os
os.makedirs("models", exist_ok=True)

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

# MacBook M1/M2/M3 FAST optimizers
from keras.optimizers.legacy import Adam, RMSprop, Adagrad, SGD
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, BatchNormalization, GlobalAveragePooling2D, Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from utils import load_lung_dataset, IMG_SIZE

print("Loading dataset...")
X, y, class_names = load_lung_dataset()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fix class imbalance
class_weights = compute_class_weight('balanced', classes=np.arange(5), y=np.argmax(y_train, axis=1))
class_weight_dict = dict(enumerate(class_weights))

datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    zoom_range=0.3, horizontal_flip=True, fill_mode='reflect'
)

def build_model():
    inputs = Input(shape=(128, 258, 1))
    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2,2)(x)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2,2)(x)

    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2,2)(x)

    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(5, activation='softmax')(x)
    return Model(inputs, outputs)

# All combinations
optimizers = {
    'Adam': Adam(learning_rate=0.001),
    'RMSprop': RMSprop(learning_rate=0.001),
    'Adagrad': Adagrad(learning_rate=0.01),
    'SGD': SGD(learning_rate=0.01, momentum=0.9)
}
losses = ['categorical_crossentropy']  # MSE is useless for classification — skip it

results = []
best_val_acc = 0
best_model_path = None

print("\nSTARTING FULL AUTOMATIC SEARCH FOR BEST MODEL\n" + "="*80)

for loss_name in losses:
    for opt_name, opt in optimizers.items():
        print(f"\nTesting → {opt_name} + {loss_name}")
        
        model = build_model()
        model.compile(optimizer=opt, loss=loss_name, metrics=['accuracy'])
        
        # Same strong settings for ALL
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            validation_data=(X_val, y_val),
            epochs=150,
            verbose=1,
            callbacks=[
                ModelCheckpoint("models/temp_best.keras", save_best_only=True, monitor='val_accuracy', mode='max', verbose=0),
                ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, min_lr=1e-8),
                EarlyStopping(patience=30, restore_best_weights=True, monitor='val_accuracy')
            ],
            class_weight=class_weight_dict
        )
        
        val_acc = max(history.history['val_accuracy'])
        train_acc = history.history['accuracy'][-1]
        
        # Evaluation
        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        y_true = np.argmax(y_val, axis=1)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        results.append({
            'Optimizer': opt_name,
            'Loss': loss_name,
            'Precision': round(p, 4),
            'Recall': round(r, 4),
            'F1': round(f1, 4),
            'Train_Acc': round(train_acc, 4),
            'Val_Acc': round(val_acc, 4)
        })
        
        print(f"Result → Val Accuracy: {val_acc:.4f}")
        
        # Automatically save the TRUE best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = "models/temp_best.keras"
            # Copy as final best
            import shutil
            shutil.copy("models/temp_best.keras", "models/best_model.keras")
            print(f"NEW BEST MODEL SAVED! → {val_acc:.4f}")

# Final results
df = pd.DataFrame(results)
print("\n" + "="*100)
print("FINAL RANKING — BEST MODEL AUTOMATICALLY SELECTED")
print(df.sort_values('Val_Acc', ascending=False).to_string(index=False))
print("="*100)
print(f"TRUE BEST MODEL SAVED → models/best_model.keras ({best_val_acc:.4f})")
print("Class names saved → models/class_names.pkl")
joblib.dump(class_names, "models/class_names.pkl")
print("Done! Now run: python predict.py")