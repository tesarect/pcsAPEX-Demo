from django.db import models
from django.conf import settings
from django.utils import timezone
import os
import uuid
import json

class SystemConfig(models.Model):
    """Model for system configuration"""
    key = models.CharField(max_length=100, unique=True)
    value = models.TextField()
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.key}: {self.value}"

    @classmethod
    def get_value(cls, key, default=None, as_type=None):
        """Get a configuration value by key"""
        try:
            config = cls.objects.get(key=key)
            value = config.value

            if as_type == bool:
                return value.lower() in ('true', 'yes', '1', 'on')
            elif as_type == int:
                return int(value)
            elif as_type == float:
                return float(value)
            elif as_type == list or as_type == dict:
                return json.loads(value)
            elif as_type:
                return as_type(value)

            return value
        except (cls.DoesNotExist, ValueError, json.JSONDecodeError):
            return default

    @classmethod
    def set_value(cls, key, value, description=None):
        """Set a configuration value"""
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        elif not isinstance(value, str):
            value = str(value)

        config, created = cls.objects.update_or_create(
            key=key,
            defaults={'value': value}
        )

        if description and created:
            config.description = description
            config.save()

        return config

class Category(models.Model):
    """Model for image categories/labels. Stores the different classes/labels("lentils") for your images
    (e.g., "raw", "broken", "polished", "unpolished")"""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name_plural = "Categories"

def image_upload_path(instance, filename):
    """Generate a unique path for uploaded images"""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('uploads', filename)

class Image(models.Model):
    """Model for storing image data. Stores information about each uploaded image and its classification status"""
    STATUS_CHOICES = [
        ('uploaded', 'Uploaded'),
        ('labeled', 'Labeled'),
        ('training', 'Used for Training'),
        ('processed', 'Processed'),
    ]

    image = models.ImageField(upload_to=image_upload_path)
    original_filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploaded')
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True, blank=True, related_name='images')
    confidence = models.FloatField(null=True, blank=True)  # For prediction confidence

    def __str__(self):
        return f"{self.original_filename} ({self.status})"

class TrainingJob(models.Model):
    """Model for tracking training jobs. Stores information about each training job, including its status,
    accuracy, and completion time"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    model_file = models.FileField(upload_to='models/', null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.name} ({self.status})"

    def save(self, *args, **kwargs):
        if self.status == 'completed' and not self.completed_at:
            self.completed_at = timezone.now()
        super().save(*args, **kwargs)

class PredictionBatch(models.Model):
    """Model for tracking batches of predictions. Stores information about each batch of predictions,
    including its status, completion time, and the training job it belongs to"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    training_job = models.ForeignKey(TrainingJob, on_delete=models.CASCADE, related_name='prediction_batches')

    def __str__(self):
        return f"{self.name} ({self.status})"

    def save(self, *args, **kwargs):
        if self.status == 'completed' and not self.completed_at:
            self.completed_at = timezone.now()
        super().save(*args, **kwargs)
