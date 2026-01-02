import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


def create_model(input_shape=(224, 224, 3), backbone='mobilenetv2', fine_tune_at=None):
    """Create a transfer learning model for binary classification with a strong backbone"""
    if backbone == 'efficientnetb0':
        base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    else:
        base = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = False
    if fine_tune_at is not None:
        for layer in base.layers[-fine_tune_at:]:
            layer.trainable = True
    inputs = layers.Input(shape=input_shape)
    aug = models.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])
    x = aug(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_training_history(history, save_plots=True, timestamp=None):
    """Plot training history and optionally save as image"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()

    if save_plots and timestamp:
        plt.savefig(f'training_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"Training plots saved as: training_plots_{timestamp}.png")

    plt.show()


def save_training_report(model, history, val_loss, val_accuracy,
                         classification_rep, confusion_mat, class_names,
                         bagtikan_count, others_count, timestamp):
    """Save comprehensive training report to files"""

    # Create reports directory if it doesn't exist
    reports_dir = 'training_reports'
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    # 1. Save model summary to text file
    model_summary_path = os.path.join(reports_dir, f'model_summary_{timestamp}.txt')
    with open(model_summary_path, 'w', encoding='utf-8') as f:
        f.write("BAGTIKAN TREE CLASSIFIER - MODEL ARCHITECTURE\n")
        f.write("=" * 50 + "\n\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # 2. Save training history to JSON
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }

    history_path = os.path.join(reports_dir, f'training_history_{timestamp}.json')
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    # 3. Save comprehensive report
    report = {
        'timestamp': timestamp,
        'dataset_info': {
            'bagtikan_images': bagtikan_count,
            'other_images': others_count,
            'total_images': bagtikan_count + others_count
        },
        'training_config': {
            'epochs_trained': len(history.history['accuracy']),
            'batch_size': 32,
            'input_shape': [224, 224, 3],
            'validation_split': 0.2
        },
        'final_metrics': {
            'validation_loss': float(val_loss),
            'validation_accuracy': float(val_accuracy),
            'best_train_accuracy': float(max(history.history['accuracy'])),
            'best_val_accuracy': float(max(history.history['val_accuracy']))
        },
        'class_names': class_names,
        'confusion_matrix': confusion_mat.tolist(),
        'classification_report_text': classification_rep
    }

    report_path = os.path.join(reports_dir, f'training_report_{timestamp}.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # 4. Save human-readable summary
    summary_path = os.path.join(reports_dir, f'summary_report_{timestamp}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("BAGTIKAN TREE CLASSIFIER - TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training Date: {timestamp}\n")
        f.write(f"Training Duration: {len(history.history['accuracy'])} epochs\n\n")

        f.write("DATASET INFORMATION:\n")
        f.write(f"- Bagtikan images: {bagtikan_count}\n")
        f.write(f"- Other tree images: {others_count}\n")
        f.write(f"- Total images: {bagtikan_count + others_count}\n")
        f.write(f"- Training/Validation split: 80/20\n\n")

        f.write("FINAL PERFORMANCE METRICS:\n")
        f.write(f"- Final Validation Accuracy: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)\n")
        f.write(f"- Final Validation Loss: {val_loss:.4f}\n")
        f.write(f"- Best Training Accuracy: {max(history.history['accuracy']):.4f}\n")
        f.write(f"- Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}\n\n")

        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write(classification_rep + "\n\n")

        f.write("CONFUSION MATRIX:\n")
        f.write("(Rows = True labels, Columns = Predicted labels)\n")
        f.write(str(confusion_mat) + "\n\n")

        f.write("CLASS MAPPING:\n")
        for class_name, index in enumerate(class_names):
            f.write(f"- {class_name}: {index}\n")

    print(f"\n📊 REPORTS SAVED SUCCESSFULLY!")
    print(f"📁 Reports directory: {reports_dir}/")
    print(f"📄 Model summary: model_summary_{timestamp}.txt")
    print(f"📊 Training history: training_history_{timestamp}.json")
    print(f"📈 Complete report: training_report_{timestamp}.json")
    print(f"📋 Summary report: summary_report_{timestamp}.txt")
    print(f"📉 Training plots: training_plots_{timestamp}.png")


def fit_temperature_scaling(probs, labels, grid=np.linspace(0.5, 5.0, 30)):
    """Calibrate probabilities via temperature scaling using NLL minimization on validation data"""
    eps = 1e-8
    labels = labels.astype(np.float32)
    logits = np.log(np.clip(probs, eps, 1.0 - eps) / np.clip(1.0 - probs, eps, 1.0))
    best_T, best_nll = 1.0, 1e9
    for T in grid:
        pT = 1.0 / (1.0 + np.exp(-logits / T))
        nll = -np.mean(labels * np.log(np.clip(pT, eps, 1.0 - eps)) + (1.0 - labels) * np.log(np.clip(1.0 - pT, eps, 1.0)))
        if nll < best_nll:
            best_nll, best_T = nll, T
    return float(best_T), float(best_nll)

def save_calibration(T, timestamp):
    """Save calibration temperature to JSON for runtime use"""
    path = f'model_calibration_{timestamp}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'temperature': T, 'timestamp': timestamp}, f, indent=2)
    # Also save a stable filename used by the app
    with open('model_calibration.json', 'w', encoding='utf-8') as f:
        json.dump({'temperature': T, 'timestamp': timestamp}, f, indent=2)
    print(f"📏 Calibration saved: {path} and model_calibration.json (T={T:.3f})")

def train_bagtikan_classifier():
    """Main function to train the bagtikan tree classifier"""

    # Generate timestamp for this training session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🚀 Starting training session: {timestamp}")

    # Define paths
    data_dir = 'img_data'
    bagtikan_dir = os.path.join(data_dir, 'bagtikan')
    others_dir = os.path.join(data_dir, 'others')

    # Check if directories exist
    if not os.path.exists(bagtikan_dir):
        print(f"❌ Error: {bagtikan_dir} does not exist!")
        return None, None
    if not os.path.exists(others_dir):
        print(f"❌ Error: {others_dir} does not exist!")
        return None, None

    # Count images in each directory
    bagtikan_count = len([f for f in os.listdir(bagtikan_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    others_count = len([f for f in os.listdir(others_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"📁 Found {bagtikan_count} bagtikan images")
    print(f"📁 Found {others_count} other tree images")
    print(f"📊 Total images: {bagtikan_count + others_count}")

    # Image parameters
    img_width, img_height = 224, 224
    batch_size = 32

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2  # 20% for validation
    )

    # Data generator for validation (no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    # Create training data generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        seed=42
    )

    # Create validation data generator
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        seed=42
    )

    # Print class indices
    print("\n🏷️  Class indices:")
    for class_name, index in train_generator.class_indices.items():
        print(f"   {class_name}: {index}")

    # Create and compile model (transfer learning)
    model = create_model(input_shape=(img_width, img_height, 3), backbone='mobilenetv2', fine_tune_at=20)
    print("\n🧠 Model architecture:")
    model.summary()

    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001
    )

    # Train the model
    print("\n🎯 Starting training...")
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Plot training history
    plot_training_history(history, save_plots=True, timestamp=timestamp)

    # Evaluate the model
    print("\n📊 Evaluating model on validation data...")
    val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
    print(f"✅ Validation Loss: {val_loss:.4f}")
    print(f"✅ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")

    # Generate predictions for classification report
    validation_generator.reset()
    probs = model.predict(validation_generator, verbose=0).flatten()
    predicted_classes = (probs > 0.5).astype(int).flatten()
    true_classes = validation_generator.classes

    # Classification report
    class_names = list(validation_generator.class_indices.keys())
    classification_rep = classification_report(true_classes, predicted_classes,
                                               target_names=class_names)
    print("\n📈 Classification Report:")
    print(classification_rep)

    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    print("\n🔍 Confusion Matrix:")
    print(cm)

    # Confidence calibration (temperature scaling)
    T, nll = fit_temperature_scaling(probs, true_classes)
    print(f"\n📏 Temperature scaling: T={T:.3f} | Val NLL={nll:.4f}")
    save_calibration(T, timestamp)

    # Save comprehensive reports
    save_training_report(model, history, val_loss, val_accuracy,
                         classification_rep, cm, class_names,
                         bagtikan_count, others_count, timestamp)

    # Save the model
    model_path_time = f'bagtikan_classifier_{timestamp}.h5'
    model.save(model_path_time)
    # Also save to the stable filename expected by the app
    model.save('bagtikan_classifier.h5')
    print(f"\n💾 Model saved as: {model_path_time} and bagtikan_classifier.h5")

    return model, history


def predict_image(model_path, image_path):
    """Function to predict if a single image is bagtikan or not"""
    print(f"🔍 Loading model from: {model_path}")
    print(f"🖼️  Analyzing image: {image_path}")

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    prediction = model.predict(img_array, verbose=0)[0][0]

    # Interpret result (assuming bagtikan=0, others=1 based on alphabetical order)
    if prediction > 0.5:
        result = "Others (Not Bagtikan)"
        confidence = prediction
    else:
        result = "Bagtikan"
        confidence = 1 - prediction

    print(f"🎯 Prediction: {result}")
    print(f"📊 Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")

    return result, confidence


def list_available_models():
    """List all available trained models"""
    models = [f for f in os.listdir('.') if f.startswith('bagtikan_classifier_') and f.endswith('.h5')]

    if models:
        print("🤖 Available trained models:")
        for i, model in enumerate(models, 1):
            print(f"   {i}. {model}")
    else:
        print("❌ No trained models found. Please train a model first.")

    return models


def list_available_reports():
    """List all available training reports"""
    reports_dir = 'training_reports'
    if not os.path.exists(reports_dir):
        print("❌ No training reports found.")
        return []

    summary_reports = [f for f in os.listdir(reports_dir) if f.startswith('summary_report_')]

    if summary_reports:
        print("📋 Available training reports:")
        for i, report in enumerate(summary_reports, 1):
            print(f"   {i}. {report}")
    else:
        print("❌ No summary reports found.")

    return summary_reports


if __name__ == "__main__":
    print("🌳 BAGTIKAN TREE CLASSIFIER")
    print("=" * 40)

    # Train the classifier
    model, history = train_bagtikan_classifier()

    if model is not None:
        print("\n" + "=" * 50)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)

        # Show available models and reports
        print("\n📁 Available resources:")
        list_available_models()
        print()
        list_available_reports()

        print("\n🔮 PREDICTION USAGE:")
        print("To predict a new image, use:")
        print("predict_image('bagtikan_classifier_TIMESTAMP.h5', 'path/to/your/image.jpg')")
        print("\n📊 VIEW REPORTS:")
        print("Check the 'training_reports/' folder for detailed analysis!")
        print("=" * 50)
    else:
        print("\n❌ Training failed. Please check your data directories.")
