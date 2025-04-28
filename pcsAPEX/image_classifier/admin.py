from django.contrib import admin
from .models import Category, Image, TrainingJob, PredictionBatch, SystemConfig

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'created_at')
    search_fields = ('name', 'description')

@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ('original_filename', 'status', 'category', 'uploaded_at', 'confidence')
    list_filter = ('status', 'category', 'uploaded_at')
    search_fields = ('original_filename',)
    readonly_fields = ('uploaded_at',)

@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ('name', 'status', 'started_at', 'completed_at', 'accuracy')
    list_filter = ('status', 'started_at')
    search_fields = ('name', 'description')
    readonly_fields = ('started_at',)

@admin.register(PredictionBatch)
class PredictionBatchAdmin(admin.ModelAdmin):
    list_display = ('name', 'status', 'created_at', 'completed_at', 'training_job')
    list_filter = ('status', 'created_at', 'training_job')
    search_fields = ('name',)
    readonly_fields = ('created_at',)

@admin.register(SystemConfig)
class SystemConfigAdmin(admin.ModelAdmin):
    list_display = ('key', 'value', 'description', 'updated_at')
    search_fields = ('key', 'value', 'description')
    readonly_fields = ('created_at', 'updated_at')
    fieldsets = (
        (None, {
            'fields': ('key', 'value', 'description')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
