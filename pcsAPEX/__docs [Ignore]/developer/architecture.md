# System Architecture

This document describes the architecture of the Image Classification System.

## Overview

The Image Classification System is built using Django as the web framework and TensorFlow for machine learning. It follows a standard Model-View-Template (MVT) architecture.

## Components

### Models

The system uses the following main models:

- **Category**: Represents a classification category/label
- **Image**: Stores information about uploaded images and their classification
- **TrainingJob**: Tracks the status and results of model training jobs
- **PredictionBatch**: Groups predictions made by a trained model
- **SystemConfig**: Stores system configuration values

### Views

The views are organized into several categories:

- **Image views**: For uploading, viewing, and labeling images
- **Category views**: For managing classification categories
- **Training views**: For creating and managing training jobs
- **Prediction views**: For processing new images with trained models
- **Configuration views**: For managing system settings

### Machine Learning

The ML components are in the `ml_utils.py` file and include:

- **Image preprocessing**: Resizing and normalizing images
- **Model creation**: Building and compiling the neural network
- **Training**: Training the model on labeled images
- **Prediction**: Using the trained model to classify new images

## Data Flow

1. Users upload images through the web interface
2. Images are stored in the media directory and metadata in the database
3. Users label images by assigning categories
4. Users create and start training jobs
5. The system trains a model using the labeled images
6. The trained model is saved to disk
7. Users can process new images using the trained model
8. Prediction results are stored in the database

## Technologies

- **Web Framework**: Django
- **Database**: SQLite (development) / PostgreSQL (production)
- **Machine Learning**: TensorFlow, Keras
- **Frontend**: Bootstrap, JavaScript
- **Task Processing**: Background threads (development) / Celery (production)

## Deployment

The system can be deployed using:

- Docker containers
- Traditional web server setup (Nginx + Gunicorn)
- Cloud platforms (AWS, GCP, Azure)

See the deployment documentation for specific instructions.
