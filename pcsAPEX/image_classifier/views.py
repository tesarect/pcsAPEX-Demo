from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.conf import settings

from .models import Category, Image, TrainingJob, PredictionBatch
from .forms import ImageUploadForm, CategoryForm, TrainingJobForm
from . import ml_utils
from .services import get_image_statistics

import os
import uuid
import threading
from datetime import datetime

def index(request):
    """Home page view"""
    total_images = Image.objects.count()
    labeled_images = Image.objects.filter(status__in=['labeled', 'training']).count()
    training_jobs = TrainingJob.objects.count()
    categories = Category.objects.count()

    context = {
        'total_images': total_images,
        'labeled_images': labeled_images,
        'training_jobs': training_jobs,
        'categories': categories,
    }

    return render(request, 'image_classifier/index.html', context)

class ImageListView(ListView):
    """View to list all images"""
    model = Image
    template_name = 'image_classifier/image_list.html'
    context_object_name = 'images'
    paginate_by = 20

    def get_queryset(self):
        queryset = super().get_queryset()
        status = self.request.GET.get('status')
        category = self.request.GET.get('category')

        if status:
            queryset = queryset.filter(status=status)
        if category:
            queryset = queryset.filter(category__id=category)

        return queryset.order_by('-uploaded_at')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = Category.objects.all()
        return context

class ImageDetailView(DetailView):
    """View to show image details"""
    model = Image
    template_name = 'image_classifier/image_detail.html'
    context_object_name = 'image'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = Category.objects.all()
        return context

def upload_image(request):
    """View to handle image uploads"""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Get the image file
            image_file = request.FILES['image']

            # Create a new image record
            new_image = Image(
                image=image_file,
                original_filename=image_file.name,
                status='uploaded'
            )
            new_image.save()

            # Check if auto-suggest is enabled using Pydantic model
            from .schemas import TrainingConfig
            config = TrainingConfig.from_system_config()

            if config.auto_suggest_labels:
                # Check if we have a trained model to suggest a label
                completed_jobs = TrainingJob.objects.filter(status='completed')
                if completed_jobs.exists():
                    latest_job = completed_jobs.order_by('-completed_at').first()

                    # Try to predict the category
                    try:
                        img_path = os.path.join(settings.MEDIA_ROOT, str(new_image.image))
                        result = ml_utils.predict_image(img_path, latest_job.id)

                        if result['success']:
                            # Only suggest if confidence is above threshold
                            if result['confidence'] >= config.min_confidence_threshold:
                                # Store the suggestion in the session for the confirmation page
                                request.session['suggested_category'] = {
                                    'image_id': new_image.id,
                                    'category_name': result['category'],
                                    'confidence': result['confidence']
                                }

                                messages.success(request, f'Image "{image_file.name}" uploaded successfully with a suggested category!')
                                return redirect('confirm_image_label', pk=new_image.id)
                    except Exception as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(f"Error suggesting category: {str(e)}")

            messages.success(request, f'Image "{image_file.name}" uploaded successfully!')
            return redirect('image_list')
    else:
        form = ImageUploadForm()

    return render(request, 'image_classifier/upload_image.html', {'form': form})

def confirm_image_label(request, pk):
    """View to confirm a suggested image label"""
    image = get_object_or_404(Image, pk=pk)

    # Get the suggestion from the session
    suggestion = request.session.get('suggested_category', {})

    if not suggestion or suggestion.get('image_id') != image.id:
        messages.warning(request, 'No label suggestion available for this image.')
        return redirect('label_image', pk=pk)

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'accept':
            # Accept the suggested category
            try:
                category, created = Category.objects.get_or_create(name=suggestion['category_name'])
                image.category = category
                image.status = 'labeled'
                image.confidence = suggestion['confidence']
                image.save()

                messages.success(request, f'Image labeled as {category.name}')

                # Clear the suggestion from the session
                if 'suggested_category' in request.session:
                    del request.session['suggested_category']

                return redirect('image_list')
            except Exception as e:
                messages.error(request, f'Error accepting suggestion: {str(e)}')

        elif action == 'reject':
            # Reject and go to manual labeling
            if 'suggested_category' in request.session:
                del request.session['suggested_category']

            return redirect('label_image', pk=pk)

    # Get all categories for the form
    categories = Category.objects.all()

    return render(request, 'image_classifier/confirm_image_label.html', {
        'image': image,
        'suggestion': suggestion,
        'categories': categories
    })

def label_image(request, pk):
    """View to label an image"""
    image = get_object_or_404(Image, pk=pk)

    if request.method == 'POST':
        category_id = request.POST.get('category')
        if category_id:
            category = get_object_or_404(Category, pk=category_id)
            image.category = category
            image.status = 'labeled'  # Always set status to 'labeled' when category is assigned
            image.save()

            messages.success(request, f'Image labeled as {category.name}')
            return redirect('image_list')

    categories = Category.objects.all()
    return render(request, 'image_classifier/label_image.html', {
        'image': image,
        'categories': categories
    })

def edit_image_label(request, pk):
    """View to edit an image label regardless of status"""
    image = get_object_or_404(Image, pk=pk)

    if request.method == 'POST':
        category_id = request.POST.get('category')
        if category_id:
            category = get_object_or_404(Category, pk=category_id)
            old_category = image.category.name if image.category else "None"
            image.category = category

            # Keep the status as 'processed' if it was already processed
            if image.status != 'processed':
                image.status = 'labeled'

            image.save()

            messages.success(request, f'Image label changed from {old_category} to {category.name}')
            return redirect('image_detail', pk=pk)

    categories = Category.objects.all()
    return render(request, 'image_classifier/edit_image_label.html', {
        'image': image,
        'categories': categories
    })

class CategoryListView(ListView):
    """View to list all categories"""
    model = Category
    template_name = 'image_classifier/category_list.html'
    context_object_name = 'categories'

class CategoryCreateView(CreateView):
    """View to create a new category"""
    model = Category
    form_class = CategoryForm
    template_name = 'image_classifier/category_form.html'
    success_url = reverse_lazy('category_list')

    def form_valid(self, form):
        messages.success(self.request, 'Category created successfully!')
        return super().form_valid(form)

class TrainingJobListView(ListView):
    """View to list all training jobs"""
    model = TrainingJob
    template_name = 'image_classifier/training_job_list.html'
    context_object_name = 'training_jobs'

    def get_queryset(self):
        return TrainingJob.objects.all().order_by('-started_at')

class TrainingJobCreateView(CreateView):
    """View to create a new training job"""
    model = TrainingJob
    form_class = TrainingJobForm
    template_name = 'image_classifier/training_job_form.html'
    success_url = reverse_lazy('training_job_list')

    def form_valid(self, form):
        messages.success(self.request, 'Training job created successfully!')
        return super().form_valid(form)

class TrainingJobDetailView(DetailView):
    """View to show training job details"""
    model = TrainingJob
    template_name = 'image_classifier/training_job_detail.html'
    context_object_name = 'training_job'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['prediction_batches'] = self.object.prediction_batches.all()
        return context

def system_config(request):
    """View to manage system configurations"""
    from .models import SystemConfig

    # Default configurations
    default_configs = {
        'min_confidence_threshold': {
            'value': '70',
            'description': 'Minimum confidence threshold (%) for auto-labeling images'
        },
        'auto_suggest_labels': {
            'value': 'false',
            'description': 'Automatically suggest labels for newly uploaded images'
        },
        'training_epochs': {
            'value': '10',
            'description': 'Number of epochs to train the model'
        },
        'batch_size': {
            'value': '32',
            'description': 'Batch size for training'
        },
        'use_gpu': {
            'value': 'auto',
            'description': 'Use GPU for training if available (auto, true, false)'
        },
        'learning_rate': {
            'value': '0.001',
            'description': 'Learning rate for model optimizer'
        }
    }

    # Create default configurations ONLY if they don't exist
    for key, config in default_configs.items():
        # Check if the config already exists
        existing_value = SystemConfig.get_value(key)
        if existing_value is None:
            # Only set the default if the config doesn't exist
            SystemConfig.set_value(key, config['value'], config['description'])
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Created default config: {key} = {config['value']}")

    if request.method == 'POST':
        # Update configurations
        for key in default_configs.keys():
            if key in request.POST:
                value = request.POST[key]
                # Log the configuration update
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Updating system config: {key} = {value}")

                # Save the configuration
                config = SystemConfig.set_value(key, value)

                # Verify the save was successful
                saved_value = SystemConfig.get_value(key)
                if saved_value != value:
                    logger.error(f"Failed to save config {key}: expected {value}, got {saved_value}")
                    messages.error(request, f'Failed to save configuration for {key}')
                else:
                    logger.info(f"Successfully saved config {key} = {value}")

        messages.success(request, 'Configuration updated successfully!')
        return redirect('system_config')

    # Get all configurations
    configs = SystemConfig.objects.all().order_by('key')

    return render(request, 'image_classifier/system_config.html', {
        'configs': configs
    })

def start_training(request, pk):
    """View to start a training job"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training job with ID {pk}")

    training_job = get_object_or_404(TrainingJob, pk=pk)
    logger.info(f"Found training job: {training_job.name} (status: {training_job.status})")

    stats = get_image_statistics()
    logger.info(f"Image statistics: {stats}")

    # Check if there are images stuck in 'training' status
    if 'training' in stats['status_counts'] and stats['status_counts']['training'] > 0:
        # Reset training images before proceeding
        from .services import reset_training_images
        reset_result = reset_training_images()
        logger.info(f"Reset {reset_result['count']} images from 'training' to 'labeled' status")
        messages.info(request, f"Reset {reset_result['count']} images that were stuck in 'training' status.")
        
        # Refresh stats after reset
        stats = get_image_statistics()
        logger.info(f"Updated image statistics after reset: {stats}")

    if training_job.status == 'pending':
        # Check if we have enough labeled images
        labeled_images = Image.objects.filter(status='labeled', category__isnull=False)
        all_images = Image.objects.all()
        logger.info(f"Found {labeled_images.count()} labeled images out of {all_images.count()} total images")
        logger.info(f"Image statuses: {stats['status_counts']}")
        
        if labeled_images.count() < 10:
            logger.warning(f"Not enough labeled images: {labeled_images.count()} < 10")
            messages.error(request, 'Not enough labeled images. Please label at least 10 images before training.')
            return redirect('training_job_detail', pk=pk)

        # Check if we have at least 2 categories
        categories = Category.objects.filter(
            id__in=labeled_images.values_list('category_id', flat=True).distinct()
        )
        logger.info(f"Found {categories.count()} distinct categories")

        if categories.count() < 2:
            logger.warning(f"Not enough categories: {categories.count()} < 2")
            messages.error(request, 'At least 2 different categories are required for training.')
            return redirect('training_job_detail', pk=pk)

        # Check GPU availability
        is_gpu_available, gpu_message, gpu_details = ml_utils.check_gpu_availability()
        logger.info(f"GPU availability check: {is_gpu_available}, {gpu_message}")

        if not is_gpu_available:
            messages.warning(request, f'GPU not available: {gpu_message} Training will be slower.')

        # Update job status to in_progress before starting the thread
        training_job.status = 'in_progress'
        training_job.save()
        logger.info(f"Updated job status to 'in_progress'")

        # Start training in a background thread
        def train_in_background():
            try:
                logger.info(f"Background thread started for job {training_job.id}")

                # Double-check that the job status is still in_progress
                job = TrainingJob.objects.get(id=training_job.id)
                if job.status != 'in_progress':
                    job.status = 'in_progress'
                    job.save()
                    logger.info(f"Re-updated job status to 'in_progress'")

                # Call the train_model function
                logger.info(f"Calling train_model for job {training_job.id}")
                result = ml_utils.train_model(training_job.id)
                logger.info(f"train_model returned: {result}")

                if not result['success']:
                    # Log the error
                    logger.error(f"Training failed: {result.get('error', 'Unknown error')}")

                    # Update the job status to failed
                    try:
                        job = TrainingJob.objects.get(id=training_job.id)
                        if job.status != 'completed':  # Don't override completed status
                            job.status = 'failed'
                            job.save()
                            logger.info(f"Updated job status to 'failed'")
                    except Exception as e:
                        logger.error(f"Failed to update job status: {str(e)}")
            except Exception as e:
                # Handle any unexpected exceptions
                import traceback
                logger.error(f"Unhandled exception in training thread: {str(e)}")
                logger.error(traceback.format_exc())

                # Update the job status to failed
                try:
                    job = TrainingJob.objects.get(id=training_job.id)
                    if job.status != 'completed':  # Don't override completed status
                        job.status = 'failed'
                        job.save()
                        logger.info(f"Updated job status to 'failed' after exception")
                except Exception as inner_e:
                    logger.error(f"Failed to update job status: {str(inner_e)}")

        # Create and start the thread
        logger.info(f"Creating background thread for job {training_job.id}")
        thread = threading.Thread(target=train_in_background)
        thread.daemon = True
        thread.start()
        logger.info(f"Background thread started for job {training_job.id}")

        messages.success(request, f'Training job "{training_job.name}" started! This may take several minutes. Check back later for results.')
    else:
        logger.warning(f"Job {training_job.id} is already {training_job.status}, cannot start training")
        messages.warning(request, f'Training job "{training_job.name}" is already {training_job.status}')

    return redirect('training_job_detail', pk=pk)

def retrain_job(request, pk):
    """View to retrain an existing job"""
    training_job = get_object_or_404(TrainingJob, pk=pk)

    # Only allow retraining of completed or failed jobs
    if training_job.status not in ['completed', 'failed']:
        messages.error(request, f'Cannot retrain job "{training_job.name}" because it is currently {training_job.status}.')
        return redirect('training_job_detail', pk=pk)

    # Reset the job status
    training_job.status = 'pending'
    training_job.completed_at = None
    training_job.accuracy = None
    training_job.save()

    messages.success(request, f'Training job "{training_job.name}" has been reset and is ready for retraining.')
    return redirect('training_job_detail', pk=pk)

def delete_job(request, pk):
    """View to delete a training job"""
    training_job = get_object_or_404(TrainingJob, pk=pk)

    # Store the name for the success message
    job_name = training_job.name

    # Delete the job
    training_job.delete()

    messages.success(request, f'Training job "{job_name}" has been deleted.')
    return redirect('training_job_list')

def process_new_images(request):
    """View to process new images using a trained model"""
    import logging
    logger = logging.getLogger(__name__)

    new_images = Image.objects.filter(status='uploaded')

    if not new_images.exists():
        messages.info(request, 'No new images to process')
        return redirect('image_list')

    completed_jobs = TrainingJob.objects.filter(status='completed')
    if not completed_jobs.exists():
        messages.warning(request, 'No completed training jobs available for processing')
        return redirect('image_list')

    # Get the latest completed training job
    latest_job = completed_jobs.order_by('-completed_at').first()

    # Create a new prediction batch
    batch = PredictionBatch.objects.create(
        name=f'Batch {datetime.now().strftime("%Y%m%d-%H%M%S")}',
        status='pending',
        training_job=latest_job
    )

    # Process images in a background thread
    def process_in_background():
        try:
            logger.info(f"Starting image processing for batch {batch.id}")
            result = ml_utils.process_images_batch(batch.id)
            if result['success']:
                logger.info(f"Successfully processed {result['processed']} images")
            else:
                logger.error(f"Error processing images: {result.get('error', 'Unknown error')}")
                # Update batch status to failed if not already updated
                try:
                    batch_obj = PredictionBatch.objects.get(id=batch.id)
                    if batch_obj.status == 'in_progress':
                        batch_obj.status = 'failed'
                        batch_obj.save()
                except Exception as e:
                    logger.error(f"Failed to update batch status: {str(e)}")
        except Exception as e:
            logger.error(f"Unhandled exception in process_in_background: {str(e)}")
            try:
                batch_obj = PredictionBatch.objects.get(id=batch.id)
                batch_obj.status = 'failed'
                batch_obj.save()
            except Exception as inner_e:
                logger.error(f"Failed to update batch status: {str(inner_e)}")

    thread = threading.Thread(target=process_in_background)
    thread.daemon = True
    thread.start()

    messages.success(request, 'Image processing started. This may take a moment. Check the image list for results.')
    return redirect('image_list')

def reset_training_images(request):
    """Reset images stuck in 'training' status back to 'labeled'"""
    from .models import Image
    training_images = Image.objects.filter(status='training')
    count = training_images.count()
    
    # Only reset images that have a category assigned
    training_images.filter(category__isnull=False).update(status='labeled')
    
    messages.success(request, f'{count} images reset from training to labeled status.')
    return redirect('image_list')

def get_training_progress(request, pk):
    """API endpoint to get training progress"""
    from django.core.cache import cache
    from django.http import JsonResponse
    
    # Get the training job
    training_job = get_object_or_404(TrainingJob, pk=pk)
    
    # Get progress from cache
    cache_key = f"training_progress_{pk}"
    progress_data = cache.get(cache_key)
    
    if not progress_data:
        # Default progress data if none is found
        progress_data = {
            'phase': 'unknown',
            'progress': 0,
            'message': 'No progress data available',
            'updated_at': None
        }
    
    # Add job status
    progress_data['job_status'] = training_job.status
    
    return JsonResponse(progress_data)
