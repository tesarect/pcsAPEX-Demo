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

import os
import uuid
from datetime import datetime

def index(request):
    """Home page view"""
    total_images = Image.objects.count()
    labeled_images = Image.objects.filter(status='labeled').count()
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
            for image_file in request.FILES.getlist('images'):
                new_image = Image(
                    image=image_file,
                    original_filename=image_file.name,
                    status='uploaded'
                )
                new_image.save()

            messages.success(request, f'{len(request.FILES.getlist("images"))} images uploaded successfully!')
            return redirect('image_list')
    else:
        form = ImageUploadForm()

    return render(request, 'image_classifier/upload_image.html', {'form': form})

def label_image(request, pk):
    """View to label an image"""
    image = get_object_or_404(Image, pk=pk)

    if request.method == 'POST':
        category_id = request.POST.get('category')
        if category_id:
            category = get_object_or_404(Category, pk=category_id)
            image.category = category
            image.status = 'labeled'
            image.save()

            messages.success(request, f'Image labeled as {category.name}')
            return redirect('image_list')

    categories = Category.objects.all()
    return render(request, 'image_classifier/label_image.html', {
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

def start_training(request, pk):
    """View to start a training job"""
    training_job = get_object_or_404(TrainingJob, pk=pk)

    if training_job.status == 'pending':
        # In a real application, this would trigger a background task
        # For now, we'll just update the status
        training_job.status = 'in_progress'
        training_job.save()

        messages.success(request, f'Training job "{training_job.name}" started!')
    else:
        messages.warning(request, f'Training job "{training_job.name}" is already {training_job.status}')

    return redirect('training_job_detail', pk=pk)

def process_new_images(request):
    """View to process new images (would be called by a scheduled task in production)"""
    # In a real application, this would be a background task
    # For now, we'll just simulate the process

    new_images = Image.objects.filter(status='uploaded')
    processed_count = 0

    if new_images.exists() and TrainingJob.objects.filter(status='completed').exists():
        latest_job = TrainingJob.objects.filter(status='completed').order_by('-completed_at').first()

        # Create a new prediction batch
        batch = PredictionBatch.objects.create(
            name=f'Batch {datetime.now().strftime("%Y%m%d-%H%M%S")}',
            status='in_progress',
            training_job=latest_job
        )

        # Process each image
        for image in new_images:
            # In a real application, this would use the trained model to predict
            # For now, we'll just randomly assign a category if available
            categories = Category.objects.all()
            if categories.exists():
                import random
                image.category = random.choice(categories)
                image.confidence = random.uniform(0.7, 0.99)
                image.status = 'processed'
                image.save()
                processed_count += 1

        batch.status = 'completed'
        batch.completed_at = datetime.now()
        batch.save()

        messages.success(request, f'Processed {processed_count} new images')
    else:
        if not new_images.exists():
            messages.info(request, 'No new images to process')
        else:
            messages.warning(request, 'No completed training jobs available for processing')

    return redirect('image_list')
