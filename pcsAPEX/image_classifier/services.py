from django.db.models import Count
from .models import Image

def get_image_statistics():
    """Get statistics about images in the system"""
    total_images = Image.objects.count()
    status_counts = dict(Image.objects.values_list('status').annotate(count=Count('id')))
    return {
        'total': total_images,
        'status_counts': status_counts
    }

def reset_training_images(image_ids=None):
    """Reset images stuck in 'training' status back to 'labeled'
    
    Args:
        image_ids: Optional list of specific image IDs to reset
        
    Returns:
        Dictionary with count of reset images and success status
    """
    from .models import Image
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        if image_ids and len(image_ids) > 0:
            # Reset specific images
            training_images = Image.objects.filter(id__in=image_ids, status='training')
            count = training_images.count()
            if count > 0:
                training_images.update(status='labeled')
            logger.info(f"Reset {count} specific images from 'training' to 'labeled' status")
        else:
            # Reset all training images that have a category assigned
            training_images = Image.objects.filter(status='training', category__isnull=False)
            count = training_images.count()
            if count > 0:
                training_images.update(status='labeled')
            logger.info(f"Reset {count} images from 'training' to 'labeled' status")
            
        return {
            'count': count,
            'success': True
        }
    except Exception as e:
        logger.error(f"Failed to reset training images: {str(e)}")
        return {
            'count': 0,
            'success': False,
            'error': str(e)
        }
