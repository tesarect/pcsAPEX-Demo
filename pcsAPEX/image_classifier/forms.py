from django import forms
from .models import Category, Image, TrainingJob

class ImageUploadForm(forms.Form):
    """Form for uploading images"""
    # For simplicity, we'll use a single file field
    image = forms.ImageField(
        help_text='Select an image to upload'
    )

class CategoryForm(forms.ModelForm):
    """Form for creating/editing categories"""
    class Meta:
        model = Category
        fields = ['name', 'description']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
        }

class TrainingJobForm(forms.ModelForm):
    """Form for creating/editing training jobs"""
    class Meta:
        model = TrainingJob
        fields = ['name', 'description']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
        }
