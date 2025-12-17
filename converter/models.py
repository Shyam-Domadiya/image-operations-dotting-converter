"""
Django models for the image dotting converter application.
"""
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.core.files.storage import default_storage
import os


class ProcessingParameters:
    """
    Data structure for image processing parameters.
    This is not a Django model but a simple data class for parameter management.
    """
    
    def __init__(self, dot_spacing=20, min_dot_radius=1, max_dot_radius=10, 
                 color_mode='black_on_white', uploaded_image=None):
        self.dot_spacing = dot_spacing
        self.min_dot_radius = min_dot_radius
        self.max_dot_radius = max_dot_radius
        self.color_mode = color_mode
        self.uploaded_image = uploaded_image
    
    def validate(self):
        """Validate parameter values."""
        errors = []
        
        if not (10 <= self.dot_spacing <= 50):
            errors.append("Dot spacing must be between 10 and 50 pixels.")
        
        if not (1 <= self.min_dot_radius <= 5):
            errors.append("Minimum dot radius must be between 1 and 5 pixels.")
        
        if not (5 <= self.max_dot_radius <= 20):
            errors.append("Maximum dot radius must be between 5 and 20 pixels.")
        
        if self.min_dot_radius >= self.max_dot_radius:
            errors.append("Minimum dot radius must be less than maximum dot radius.")
        
        if self.color_mode not in ['black_on_white', 'white_on_black']:
            errors.append("Color mode must be 'black_on_white' or 'white_on_black'.")
        
        return errors
    
    def is_valid(self):
        """Check if parameters are valid."""
        return len(self.validate()) == 0
    
    def to_dict(self):
        """Convert parameters to dictionary."""
        return {
            'dot_spacing': self.dot_spacing,
            'min_dot_radius': self.min_dot_radius,
            'max_dot_radius': self.max_dot_radius,
            'color_mode': self.color_mode,
            'uploaded_image': self.uploaded_image
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create ProcessingParameters from dictionary."""
        return cls(
            dot_spacing=data.get('dot_spacing', 20),
            min_dot_radius=data.get('min_dot_radius', 1),
            max_dot_radius=data.get('max_dot_radius', 10),
            color_mode=data.get('color_mode', 'black_on_white'),
            uploaded_image=data.get('uploaded_image')
        )


class ImageMetadata:
    """
    Data structure for image metadata.
    This is not a Django model but a simple data class for metadata management.
    """
    
    def __init__(self, original_width=0, original_height=0, grid_cols=0, 
                 grid_rows=0, output_filename=""):
        self.original_width = original_width
        self.original_height = original_height
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.output_filename = output_filename
    
    def to_dict(self):
        """Convert metadata to dictionary."""
        return {
            'original_width': self.original_width,
            'original_height': self.original_height,
            'grid_cols': self.grid_cols,
            'grid_rows': self.grid_rows,
            'output_filename': self.output_filename
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create ImageMetadata from dictionary."""
        return cls(
            original_width=data.get('original_width', 0),
            original_height=data.get('original_height', 0),
            grid_cols=data.get('grid_cols', 0),
            grid_rows=data.get('grid_rows', 0),
            output_filename=data.get('output_filename', "")
        )


# Optional Django models for persistent storage (if needed in the future)

class ProcessingJob(models.Model):
    """
    Optional Django model for storing processing job information.
    Currently not used but available for future enhancements.
    """
    
    COLOR_MODE_CHOICES = [
        ('black_on_white', 'Black dots on white background'),
        ('white_on_black', 'White dots on black background'),
    ]
    
    # File paths
    input_image = models.ImageField(upload_to='input/', null=True, blank=True)
    output_image = models.ImageField(upload_to='output/', null=True, blank=True)
    
    # Processing parameters
    dot_spacing = models.IntegerField(
        default=20,
        validators=[MinValueValidator(10), MaxValueValidator(50)]
    )
    min_dot_radius = models.IntegerField(
        default=1,
        validators=[MinValueValidator(1), MaxValueValidator(5)]
    )
    max_dot_radius = models.IntegerField(
        default=10,
        validators=[MinValueValidator(5), MaxValueValidator(20)]
    )
    color_mode = models.CharField(
        max_length=20,
        choices=COLOR_MODE_CHOICES,
        default='black_on_white'
    )
    
    # Metadata
    original_width = models.IntegerField(default=0)
    original_height = models.IntegerField(default=0)
    grid_cols = models.IntegerField(default=0)
    grid_rows = models.IntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Status
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending'
    )
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Processing Job {self.id} - {self.status}"
    
    def get_processing_parameters(self):
        """Get ProcessingParameters object from model data."""
        return ProcessingParameters(
            dot_spacing=self.dot_spacing,
            min_dot_radius=self.min_dot_radius,
            max_dot_radius=self.max_dot_radius,
            color_mode=self.color_mode,
            uploaded_image=self.input_image
        )
    
    def get_image_metadata(self):
        """Get ImageMetadata object from model data."""
        return ImageMetadata(
            original_width=self.original_width,
            original_height=self.original_height,
            grid_cols=self.grid_cols,
            grid_rows=self.grid_rows,
            output_filename=os.path.basename(self.output_image.name) if self.output_image else ""
        )
