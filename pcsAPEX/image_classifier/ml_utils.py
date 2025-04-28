"""
Machine learning utilities for image classification
"""
import os
import numpy as np
import cv2
import pickle
import logging
from pathlib import Path
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def check_gpu_availability():
    """
    Check if GPU is available for TensorFlow

    Returns:
        tuple: (is_gpu_available, message)
    """
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Try to create a small test tensor on GPU
            with tf.device('/GPU:0'):
                test_tensor = tf.random.normal([1000, 1000])
                result = tf.reduce_sum(test_tensor)

            gpu_info = []
            for gpu in gpus:
                gpu_info.append(str(gpu))

            return True, f"GPU is available: {', '.join(gpu_info)}"
        else:
            return False, "No GPU devices found. Training will use CPU only (slower)."
    except Exception as e:
        return False, f"Error checking GPU: {str(e)}. Training will use CPU only (slower)."

# Image dimensions for model input
IMG_WIDTH = 224
IMG_HEIGHT = 224

def preprocess_image(image_path):
    """
    Preprocess an image for model input

    Args:
        image_path: Path to the image file

    Returns:
        Preprocessed image as numpy array
    """
    # Read and resize image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # Preprocess for MobileNetV2
    img = preprocess_input(img)

    return img

def create_model(num_classes):
    """
    Create a transfer learning model based on MobileNetV2

    Args:
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    # Use MobileNetV2 as base model (smaller and faster than other models)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    )

    # Freeze the base model layers
    base_model.trainable = False

    # Create new model on top
    model = Sequential([
        base_model,
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(training_job_id):
    """
    Train a model using labeled images

    Args:
        training_job_id: ID of the training job

    Returns:
        Dictionary with training results
    """
    from .models import TrainingJob, Image, Category, SystemConfig
    import traceback

    # Get the training job
    try:
        job = TrainingJob.objects.get(id=training_job_id)
        # Note: Status should already be set to 'in_progress' by the view
        # This is a safety check
        if job.status != 'in_progress':
            job.status = 'in_progress'
            job.save()

        logger.info(f"Starting training job {job.id}: {job.name}")

        # Get configuration values
        use_gpu_config = SystemConfig.get_value('use_gpu', 'auto')
        training_epochs = SystemConfig.get_value('training_epochs', '10', as_type=int)
        batch_size = SystemConfig.get_value('batch_size', '32', as_type=int)

        # Check GPU availability
        is_gpu_available, gpu_message = check_gpu_availability()

        # Determine if we should use GPU based on config
        use_gpu = False
        if use_gpu_config == 'auto':
            use_gpu = is_gpu_available
        elif use_gpu_config == 'true':
            use_gpu = True

        if use_gpu and not is_gpu_available:
            logger.warning("GPU usage requested but no GPU is available. Falling back to CPU.")
            use_gpu = False

        logger.info(f"GPU check: {gpu_message}")
        logger.info(f"Using GPU: {use_gpu}")
        logger.info(f"Training with epochs: {training_epochs}, batch size: {batch_size}")

        # Get labeled images
        labeled_images = Image.objects.filter(status='labeled', category__isnull=False)

        if labeled_images.count() < 10:
            error_msg = 'Not enough labeled images (minimum 10 required)'
            logger.error(error_msg)
            job.status = 'failed'
            job.save()
            return {
                'success': False,
                'error': error_msg
            }

        # Get categories
        categories = Category.objects.filter(
            id__in=labeled_images.values_list('category_id', flat=True).distinct()
        )

        if categories.count() < 2:
            error_msg = 'At least 2 categories are required for training'
            logger.error(error_msg)
            job.status = 'failed'
            job.save()
            return {
                'success': False,
                'error': error_msg
            }

        logger.info(f"Starting training with {labeled_images.count()} images in {categories.count()} categories")

        # Prepare data
        X = []  # Images
        y = []  # Labels

        for image in labeled_images:
            try:
                img_path = os.path.join(settings.MEDIA_ROOT, str(image.image))
                img = preprocess_image(img_path)
                X.append(img)
                y.append(image.category.name)

                # Update image status
                image.status = 'training'
                image.save()
            except Exception as e:
                logger.error(f"Error processing image {image.id}: {str(e)}")

        if not X:
            error_msg = 'Failed to process any images'
            logger.error(error_msg)
            job.status = 'failed'
            job.save()
            return {
                'success': False,
                'error': error_msg
            }

        # Convert to numpy arrays
        X = np.array(X)

        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Get number of classes
        num_classes = len(label_encoder.classes_)
        logger.info(f"Number of classes: {num_classes}")

        # Convert to one-hot encoding
        y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)

        logger.info(f"Data shapes: X={X.shape}, y={y_onehot.shape}")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_onehot, test_size=0.2, random_state=42
        )

        # Create model
        model = create_model(num_classes)

        # Data augmentation for training
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        try:
            # Train model
            logger.info("Starting model training...")
            history = model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                epochs=training_epochs,
                validation_data=(X_val, y_val),
                verbose=1
            )

            # Evaluate model
            logger.info("Evaluating model...")
            _, accuracy = model.evaluate(X_val, y_val, verbose=0)

            # Save model and label encoder
            models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
            os.makedirs(models_dir, exist_ok=True)

            model_path = os.path.join(models_dir, f'model_{job.id}.h5')
            encoder_path = os.path.join(models_dir, f'encoder_{job.id}.pkl')

            logger.info(f"Saving model to {model_path}")
            model.save(model_path)

            with open(encoder_path, 'wb') as f:
                pickle.dump(label_encoder, f)

            # Update job
            job.status = 'completed'
            job.accuracy = float(accuracy * 100)
            job.model_file = os.path.join('models', f'model_{job.id}.h5')
            job.save()

            # Reset image status
            labeled_images.update(status='labeled')

            logger.info(f"Training completed with accuracy: {accuracy * 100:.2f}%")

            return {
                'success': True,
                'accuracy': float(accuracy * 100),
                'model_path': model_path,
                'encoder_path': encoder_path
            }
        except Exception as e:
            error_msg = f"Error during model training: {str(e)}"
            logger.error(error_msg)
            job.status = 'failed'
            job.save()
            # Reset image status
            labeled_images.update(status='labeled')
            return {
                'success': False,
                'error': error_msg
            }

    except Exception as e:
        error_msg = f"Training error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        try:
            job = TrainingJob.objects.get(id=training_job_id)
            job.status = 'failed'
            job.save()
            logger.info(f"Updated job {job.id} status to 'failed'")
        except Exception as inner_e:
            logger.error(f"Failed to update job status: {str(inner_e)}")
            logger.error(traceback.format_exc())

        return {
            'success': False,
            'error': error_msg
        }

def predict_image(image_path, training_job_id):
    """
    Predict the category of an image using a trained model

    Args:
        image_path: Path to the image file
        training_job_id: ID of the training job to use

    Returns:
        Dictionary with prediction results
    """
    from .models import TrainingJob

    try:
        # Get the training job
        job = TrainingJob.objects.get(id=training_job_id)

        if job.status != 'completed' or not job.model_file:
            return {
                'success': False,
                'error': 'Training job not completed or model not available'
            }

        # Load model and encoder
        model_path = os.path.join(settings.MEDIA_ROOT, str(job.model_file))
        encoder_path = os.path.join(settings.MEDIA_ROOT, 'models', f'encoder_{job.id}.pkl')

        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            return {
                'success': False,
                'error': 'Model files not found'
            }

        model = load_model(model_path)

        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)

        # Preprocess image
        img = preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img)[0]
        predicted_class_index = np.argmax(predictions)
        confidence = float(predictions[predicted_class_index])

        # Get class name
        predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]

        return {
            'success': True,
            'category': predicted_class,
            'confidence': confidence * 100
        }

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def process_images_batch(batch_id):
    """
    Process a batch of images using a trained model

    Args:
        batch_id: ID of the prediction batch

    Returns:
        Dictionary with processing results
    """
    from .models import PredictionBatch, Image, Category

    try:
        # Get the batch
        batch = PredictionBatch.objects.get(id=batch_id)
        batch.status = 'in_progress'
        batch.save()

        # Get the training job
        job = batch.training_job

        if job.status != 'completed':
            batch.status = 'failed'
            batch.save()
            return {
                'success': False,
                'error': 'Training job not completed'
            }

        # Get unprocessed images
        new_images = Image.objects.filter(status='uploaded')

        if not new_images.exists():
            batch.status = 'completed'
            batch.save()
            return {
                'success': True,
                'processed': 0,
                'message': 'No images to process'
            }

        processed_count = 0

        # Process each image
        for image in new_images:
            try:
                img_path = os.path.join(settings.MEDIA_ROOT, str(image.image))

                # Make prediction
                result = predict_image(img_path, job.id)

                if result['success']:
                    # Get or create category
                    category, _ = Category.objects.get_or_create(name=result['category'])

                    # Update image
                    image.category = category
                    image.confidence = result['confidence']
                    image.status = 'processed'
                    image.save()

                    processed_count += 1
            except Exception as e:
                print(f"Error processing image {image.id}: {str(e)}")

        # Update batch
        batch.status = 'completed'
        batch.save()

        return {
            'success': True,
            'processed': processed_count
        }

    except Exception as e:
        print(f"Batch processing error: {str(e)}")
        try:
            batch.status = 'failed'
            batch.save()
        except:
            pass

        return {
            'success': False,
            'error': str(e)
        }
