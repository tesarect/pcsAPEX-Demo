"""
Machine learning utilities for image classification
"""

import os
import numpy as np
import cv2
import pickle
import logging
import traceback
from pathlib import Path
from datetime import datetime

from django.conf import settings
from django.core.cache import cache

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Import TensorFlow initialization module first and ensure it's initialized
from .tf_config import configure_tensorflow as initialize_tensorflow
# Make sure TensorFlow is properly initialized
tf_initialized = initialize_tensorflow()

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam

# Import local modules
from .models import TrainingJob, Image, Category, SystemConfig, PredictionBatch
from .services import reset_training_images as service_reset_training_images

# Import Pydantic schemas
from .schemas import TrainingConfig, PredictionResult, TrainingResult

# Get configuration using Pydantic model
training_settings = TrainingConfig.from_system_config()

# Set up logging
logger = logging.getLogger(__name__)

# Force the logger to write to both console and file
# This ensures we get logs even if Django's settings aren't properly applied
# [TODO: remove any local log handler if being used. only centralsystem had to be followed]
file_handler = logging.FileHandler('/home/lorise/.projects/demo_django/demo_django/pcsAPEX/logs/ml_training.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add handlers if they don't exist already
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(file_handler)
if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(console_handler)

# Set logger level to DEBUG to capture everything
logger.setLevel(logging.DEBUG)

# Log a test message to verify logging is working
logger.debug("ML utils module loaded and logging initialized")

def check_gpu_availability():
    """
    Check if GPU is available for TensorFlow

    Returns:
        tuple: (is_gpu_available, message, gpu_details)
    """
    try:
        # First check if TensorFlow can see any GPUs
        gpus = tf.config.list_physical_devices('GPU')

        if not gpus:
            logger.warning("No GPU devices found by TensorFlow")
            return False, "No GPU devices found. Training will use CPU only (slower).", {}

        # Get GPU details
        gpu_details = {}
        gpu_info = []

        for i, gpu in enumerate(gpus):
            gpu_info.append(str(gpu))

            # Try to get more detailed information about the GPU
            try:
                # This is a more reliable way to check if the GPU is actually usable
                with tf.device(f'/GPU:{i}'):
                    # Create a small test tensor
                    test_tensor = tf.random.normal([100, 100])
                    # Force execution
                    result = float(tf.reduce_sum(test_tensor))

                    # If we get here, the GPU is usable
                    gpu_details[f'gpu_{i}'] = {
                        'name': gpu.name,
                        'device': f'/GPU:{i}',
                        'test_passed': True
                    }
            except Exception as gpu_e:
                logger.warning(f"GPU {i} detected but not usable: {str(gpu_e)}")
                gpu_details[f'gpu_{i}'] = {
                    'name': gpu.name,
                    'device': f'/GPU:{i}',
                    'test_passed': False,
                    'error': str(gpu_e)
                }

        # Check if any GPU is actually usable
        usable_gpus = [details for details in gpu_details.values() if details.get('test_passed', False)]

        if usable_gpus:
            logger.info(f"Found {len(usable_gpus)} usable GPU(s): {', '.join([g.get('name', 'Unknown') for g in usable_gpus])}")
            return True, f"GPU is available: {', '.join(gpu_info)}", gpu_details
        else:
            logger.warning("GPUs detected but none are usable")
            return False, "GPUs detected but not usable. Training will use CPU only (slower).", gpu_details

    except Exception as e:
        logger.error(f"Error checking GPU: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False, f"Error checking GPU: {str(e)}. Training will use CPU only (slower).", {}

# Image dimensions for model input
IMG_WIDTH = 224
IMG_HEIGHT = 224

def update_training_progress(job_id, phase, progress, message):
    """
    Update the training progress for a job
    
    Args:
        job_id: ID of the training job
        phase: Current training phase (e.g., 'data_prep', 'training', 'evaluation')
        progress: Progress percentage (0-100)
        message: Status message
    """
    cache_key = f"training_progress_{job_id}"
    progress_data = {
        'phase': phase,
        'progress': progress,
        'message': message,
        'updated_at': datetime.now().isoformat()
    }
    # Store in cache for 1 hour
    cache.set(cache_key, progress_data, 3600)
    logger.info(f"Progress update for job {job_id}: {progress}% - {message}")

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

    # Get current training settings
    current_settings = TrainingConfig.from_system_config()

    # Compile model with learning rate from settings
    model.compile(
        optimizer=Adam(learning_rate=current_settings.learning_rate),
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
    # Track which images we change to 'training' status
    training_image_ids = []
    
    try:
        # Get the training job
        job = TrainingJob.objects.get(id=training_job_id)
        if job.status != 'in_progress':
            job.status = 'in_progress'
            job.save()

        logger.info(f"Starting training job {job.id}: {job.name}")
        update_training_progress(job.id, 'init', 5, "Initializing training job")
        
        # Configure GPU and training parameters
        try:
            update_training_progress(job.id, 'config', 10, "Configuring training environment")
            training_config = _configure_training_environment(job)
        except Exception as e:
            logger.error(f"Error configuring training environment: {str(e)}")
            job.status = 'failed'
            job.save()
            return TrainingResult(success=False, error=str(e)).model_dump()
        
        # Validate training data
        try:
            update_training_progress(job.id, 'validation', 15, "Validating training data")
            validation_result = _validate_training_data(job)
            if not validation_result['success']:
                return TrainingResult(success=False, error=validation_result['error']).model_dump()
            labeled_images = validation_result['labeled_images']
        except Exception as e:
            logger.error(f"Error validating training data: {str(e)}")
            job.status = 'failed'
            job.save()
            return TrainingResult(success=False, error=str(e)).model_dump()
        
        # Prepare training data
        try:
            update_training_progress(job.id, 'data_prep', 25, "Preparing training data")
            data_result = _prepare_training_data(job, labeled_images, training_image_ids)
            if not data_result['success']:
                return TrainingResult(success=False, error=data_result['error']).model_dump()
            X = data_result['X']
            y = data_result['y']
            X_val = data_result['X_val']
            y_val = data_result['y_val']
            label_encoder = data_result['label_encoder']
            labeled_images = data_result.get('labeled_images', labeled_images)
            update_training_progress(job.id, 'data_prep', 35, "Data preparation complete")
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            _reset_training_images(training_image_ids)
            job.status = 'failed'
            job.save()
            return TrainingResult(success=False, error=str(e)).model_dump()
        
        # Train and save model
        try:
            update_training_progress(job.id, 'training', 40, "Starting model training")
            return _execute_training(job, X, y, label_encoder, training_image_ids, 
                                    X_val=X_val, y_val=y_val, labeled_images=labeled_images)
        except Exception as e:
            error_msg = f"Error during model training: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            _reset_training_images(training_image_ids)
            job.status = 'failed'
            job.save()
            return TrainingResult(success=False, error=error_msg).model_dump()
            
    except Exception as e:
        error_msg = f"Training error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        try:
            job = TrainingJob.objects.get(id=training_job_id)
            job.status = 'failed'
            job.save()
        except Exception as inner_e:
            logger.error(f"Failed to update job status: {str(inner_e)}")
        
        # Reset any images in training status
        _reset_training_images(training_image_ids)
        
        return TrainingResult(
            success=False,
            error=error_msg
        ).model_dump()

def _reset_training_images(image_ids=None):
    """Reset images from training status to labeled status"""
    # Use the service function to reset images
    result = service_reset_training_images(image_ids)
    
    if not result['success']:
        logger.error(f"Failed to reset training images: {result.get('error', 'Unknown error')}")

def _configure_training_environment(job):
    """Configure GPU and training parameters"""
    # Get configuration using Pydantic model
    current_settings = TrainingConfig.from_system_config()
    
    # Check GPU availability
    is_gpu_available, gpu_message, gpu_details = check_gpu_availability()

    # Determine if we should use GPU based on config
    use_gpu = False
    if current_settings.use_gpu == "auto":
        use_gpu = is_gpu_available
    elif current_settings.use_gpu is True:
        use_gpu = True

    if use_gpu and not is_gpu_available:
        logger.warning("GPU usage requested but no GPU is available. Falling back to CPU.")
        use_gpu = False

    # Set TensorFlow to use or not use GPU
    if use_gpu:
        # Allow memory growth to avoid OOM errors
        for gpu in tf.config.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Enabled memory growth for {gpu}")
            except Exception as e:
                logger.warning(f"Could not set memory growth for {gpu}: {str(e)}")

        # Log GPU details
        logger.info(f"GPU details: {gpu_details}")
    else:
        # Force CPU usage by hiding GPU devices
        tf.config.set_visible_devices([], 'GPU')
        logger.info("Disabled GPU devices for this training session")

    logger.info(f"GPU check: {gpu_message}")
    logger.info(f"Using GPU: {use_gpu}")
    logger.info(f"Training with epochs: {current_settings.training_epochs}, batch size: {current_settings.batch_size}")

    # Save GPU usage to the training job for UI display
    job.gpu_used = use_gpu
    job.save()
    
    # Return the configuration as a dictionary for backward compatibility
    return {
        'use_gpu': use_gpu,
        'training_epochs': current_settings.training_epochs,
        'batch_size': current_settings.batch_size,
        'learning_rate': current_settings.learning_rate
    }

def _validate_training_data(job):
    """Validate that we have sufficient data for training"""
    
    # Get labeled images
    labeled_images = Image.objects.filter(status='labeled', category__isnull=False)
    logger.info(f"Found {labeled_images.count()} labeled images")

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
    logger.info(f"Found {categories.count()} distinct categories")

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
    
    return {
        'success': True,
        'labeled_images': labeled_images,
        'categories': categories
    }

def _prepare_training_data(job, labeled_images, training_image_ids):
    """Prepare training data from labeled images"""
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
            training_image_ids.append(image.id)
        except Exception as e:
            logger.error(f"Error processing image {image.id}: {str(e)}")

    if not X:
        error_msg = 'Failed to process any images'
        logger.error(error_msg)
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

    return {
        'success': True,
        'X': X_train,
        'y': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'label_encoder': label_encoder,
        'labeled_images': labeled_images  # Pass labeled_images back
    }

def _execute_training(job, X, y, label_encoder, training_image_ids, X_val=None, y_val=None, labeled_images=None):
    """Train the model and save the results"""
    # Get training configuration
    try:
        # Get fresh configuration using Pydantic model
        current_settings = TrainingConfig.from_system_config()
        
        # Create a training parameters dictionary for this function
        training_params = {
            'training_epochs': current_settings.training_epochs,
            'batch_size': current_settings.batch_size,
            'use_gpu': job.gpu_used,
            'learning_rate': current_settings.learning_rate
        }
        
        logger.info(f"Training with config: epochs={training_params['training_epochs']}, batch_size={training_params['batch_size']}, gpu={training_params['use_gpu']}")
    except Exception as e:
        logger.error(f"Error getting training configuration: {str(e)}")
        # Fallback to default values
        training_params = {
            'training_epochs': 10,
            'batch_size': 32,
            'use_gpu': False,
            'learning_rate': 0.001
        }
        logger.info("Using fallback configuration")
    
    # Create model
    update_training_progress(job.id, 'model_creation', 45, "Creating model architecture")
    model = create_model(len(label_encoder.classes_))

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
        # Set optimizer with learning rate from config
        optimizer = Adam(learning_rate=training_params['learning_rate'])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Custom callback to track progress
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                progress = 50 + int((epoch / training_params['training_epochs']) * 40)
                update_training_progress(job.id, 'training', progress, 
                                        f"Training epoch {epoch+1}/{training_params['training_epochs']}")
                
            def on_epoch_end(self, epoch, logs=None):
                progress = 50 + int(((epoch+1) / training_params['training_epochs']) * 40)
                acc = logs.get('accuracy', 0) * 100
                val_acc = logs.get('val_accuracy', 0) * 100
                update_training_progress(job.id, 'training', progress, 
                                        f"Completed epoch {epoch+1}/{training_params['training_epochs']} - Accuracy: {acc:.2f}%")
        
        # Train model
        logger.info("Starting model training...")
        update_training_progress(job.id, 'training', 50, "Starting model training")
        history = model.fit(
            datagen.flow(X, y, batch_size=training_params['batch_size']),
            epochs=training_params['training_epochs'],
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=[ProgressCallback()]
        )

        # Evaluate model
        logger.info("Evaluating model...")
        update_training_progress(job.id, 'evaluation', 90, "Evaluating model")
        _, accuracy = model.evaluate(X_val, y_val, verbose=0)

        # Save model and label encoder
        update_training_progress(job.id, 'saving', 95, "Saving model")
        models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, f'model_{job.id}.h5')
        encoder_path = os.path.join(models_dir, f'encoder_{job.id}.pkl')

        logger.info(f"Saving model to {model_path}")
        model.save(model_path)

        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)

        # Update job
        update_training_progress(job.id, 'completing', 98, "Finalizing training job")
        job.status = 'completed'
        job.accuracy = float(accuracy * 100)
        job.model_file = os.path.join('models', f'model_{job.id}.h5')
        job.save()

        # Reset image status
        if labeled_images:
            labeled_images.update(status='labeled')
        else:
            # Fallback: reset all training images
            Image.objects.filter(id__in=training_image_ids).update(status='labeled')

        logger.info(f"Training completed with accuracy: {accuracy * 100:.2f}%")
        update_training_progress(job.id, 'completed', 100, f"Training completed with accuracy: {accuracy * 100:.2f}%")

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
        if labeled_images:
            labeled_images.update(status='labeled')
        else:
            # Fallback: reset all training images
            Image.objects.filter(id__in=training_image_ids).update(status='labeled')
            
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
    from .schemas import PredictionResult
    
    try:
        # Load model and label encoder
        model, label_encoder = load_model_and_encoder(training_job_id)
        if not model or not label_encoder:
            return PredictionResult(
                success=False,
                error="Failed to load model or label encoder"
            ).model_dump()

        # Preprocess image
        img = preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img)[0]
        predicted_class_index = np.argmax(predictions)
        confidence = float(predictions[predicted_class_index])

        # Get class name
        predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]

        return PredictionResult(
            success=True,
            category=predicted_class,
            confidence=confidence * 100
        ).model_dump()

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return PredictionResult(
            success=False,
            error=str(e)
        ).model_dump()

def process_images_batch(batch_id):
    """
    Process a batch of images using a trained model

    Args:
        batch_id: ID of the prediction batch

    Returns:
        Dictionary with processing results
    """

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

def update_training_progress(job_id, phase, progress, message):
    """
    Update the training progress for a job
    
    Args:
        job_id: ID of the training job
        phase: Current training phase (e.g., 'data_prep', 'training', 'evaluation')
        progress: Progress percentage (0-100)
        message: Status message
    """
    cache_key = f"training_progress_{job_id}"
    progress_data = {
        'phase': phase,
        'progress': progress,
        'message': message,
        'updated_at': datetime.now().isoformat()
    }
    # Store in cache for 1 hour
    cache.set(cache_key, progress_data, 3600)
    logger.info(f"Progress update for job {job_id}: {progress}% - {message}")
