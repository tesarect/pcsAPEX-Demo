# Getting Started with the Image Classification System

This guide will help you get started with the Image Classification System.

## Overview

The Image Classification System is a web-based application that allows you to:

1. Upload and manage images
2. Create and manage categories for classification
3. Train machine learning models to classify images
4. Process new images using trained models

## Quick Start

### 1. Upload Images

1. Click on "Upload Images" in the navigation menu
2. Select one or more images to upload
3. Click "Upload" to add the images to the system

### 2. Create Categories

1. Click on "Categories" in the navigation menu
2. Click "Create Category"
3. Enter a name and optional description for the category
4. Click "Save" to create the category

### 3. Label Images

1. Click on "Images" in the navigation menu
2. For each unlabeled image, click "Label Image"
3. Select a category from the dropdown menu
4. Click "Save Label" to assign the category to the image

### 4. Train a Model

1. Click on "Training Jobs" in the navigation menu
2. Click "Create Training Job"
3. Enter a name and optional description for the job
4. Click "Create" to create the job
5. On the job detail page, click "Start Training"
6. Wait for the training to complete

### 5. Process New Images

1. Upload new images that you want to classify
2. Click "Process New Images" in the navigation menu
3. The system will use the latest trained model to classify the images

## System Configuration

You can configure various aspects of the system by clicking on "System Config" in the navigation menu. This includes:

- Minimum confidence threshold for auto-labeling
- Whether to auto-suggest labels for newly uploaded images
- Number of training epochs
- Batch size for training
- GPU usage settings
