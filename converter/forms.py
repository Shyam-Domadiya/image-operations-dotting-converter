"""
Django forms for the image dotting converter application.
"""
from django import forms
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator
import os


class ImageUploadForm(forms.Form):
    """Form for uploading images with file format validation."""
    
    image = forms.ImageField(
        validators=[FileExtensionValidator(allowed_extensions=[
            'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'webp', 'ico', 'psd', 'svg'
        ])],
        help_text="Upload any image file format (max 10MB, any pixel size supported)",
        widget=forms.ClearableFileInput(attrs={
            'accept': 'image/*',
            'class': 'form-control'
        })
    )
    
    def clean_image(self):
        """Validate uploaded image file with enhanced dimension and size checks."""
        image = self.cleaned_data.get('image')
        
        if not image:
            raise ValidationError("Please select an image file.")
        
        # Check file extension
        name, ext = os.path.splitext(image.name.lower())
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.ico', '.psd', '.svg']
        if ext not in allowed_extensions:
            raise ValidationError("Unsupported file format. Please upload a valid image file.")
        
        # Check MIME type - allow all image types
        if not image.content_type.startswith('image/'):
            raise ValidationError("Invalid file format. Please upload a valid image file.")
        
        # Check file size (limit to 10MB)
        if image.size > 10 * 1024 * 1024:
            raise ValidationError("File size must be less than 10MB. Please compress your image or choose a smaller file.")
        
        # Validate image dimensions for optimal processing
        try:
            from PIL import Image as PILImage
            image.seek(0)  # Reset file pointer
            
            with PILImage.open(image) as img:
                width, height = img.size
                
                # Check minimum dimensions
                if width < 10 or height < 10:
                    raise ValidationError("Image dimensions are too small. Minimum size is 10x10 pixels.")
                
                # No maximum dimension restrictions - support any pixel size
                # Note: Very large images may take longer to process
            
            image.seek(0)  # Reset file pointer for later use
            
        except ValidationError:
            raise  # Re-raise validation errors
        except Exception as e:
            raise ValidationError("Unable to process the uploaded image. Please ensure it's a valid image file.")
        
        return image


class ProcessingParametersForm(forms.Form):
    """Form for configuring image processing parameters."""
    
    COLOR_MODE_CHOICES = [
        ('black_on_white', 'Black dots on white background'),
        ('white_on_black', 'White dots on black background'),
    ]
    
    dot_spacing = forms.IntegerField(
        min_value=10,
        max_value=50,
        initial=20,
        help_text="Distance between dots (10-50 pixels)",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '1'
        })
    )
    
    min_dot_radius = forms.IntegerField(
        min_value=1,
        max_value=5,
        initial=1,
        help_text="Minimum dot size (1-5 pixels)",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '1'
        })
    )
    
    max_dot_radius = forms.IntegerField(
        min_value=5,
        max_value=20,
        initial=10,
        help_text="Maximum dot size (5-20 pixels)",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '1'
        })
    )
    
    color_mode = forms.ChoiceField(
        choices=COLOR_MODE_CHOICES,
        initial='black_on_white',
        help_text="Choose dot and background colors",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    def clean(self):
        """Validate parameter relationships."""
        cleaned_data = super().clean()
        min_radius = cleaned_data.get('min_dot_radius')
        max_radius = cleaned_data.get('max_dot_radius')
        
        if min_radius and max_radius and min_radius >= max_radius:
            raise ValidationError("Minimum dot radius must be less than maximum dot radius.")
        
        return cleaned_data
    
    def clean_dot_spacing(self):
        """Validate dot spacing parameter."""
        dot_spacing = self.cleaned_data.get('dot_spacing')
        
        if dot_spacing is not None:
            if dot_spacing < 10 or dot_spacing > 50:
                raise ValidationError("Dot spacing must be between 10 and 50 pixels.")
        
        return dot_spacing
    
    def clean_min_dot_radius(self):
        """Validate minimum dot radius parameter."""
        min_radius = self.cleaned_data.get('min_dot_radius')
        
        if min_radius is not None:
            if min_radius < 1 or min_radius > 5:
                raise ValidationError("Minimum dot radius must be between 1 and 5 pixels.")
        
        return min_radius
    
    def clean_max_dot_radius(self):
        """Validate maximum dot radius parameter."""
        max_radius = self.cleaned_data.get('max_dot_radius')
        
        if max_radius is not None:
            if max_radius < 5 or max_radius > 20:
                raise ValidationError("Maximum dot radius must be between 5 and 20 pixels.")
        
        return max_radius


class CombinedUploadForm(forms.Form):
    """Combined form for image upload and parameter configuration."""
    
    # Image upload field
    image = forms.ImageField(
        validators=[FileExtensionValidator(allowed_extensions=[
            'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'webp', 'ico', 'psd', 'svg'
        ])],
        help_text="Upload any image file format (max 10MB, any pixel size supported)",
        widget=forms.ClearableFileInput(attrs={
            'accept': 'image/*',
            'class': 'form-control'
        })
    )
    
    # Processing parameters
    COLOR_MODE_CHOICES = [
        ('black_on_white', 'Black dots on white background'),
        ('white_on_black', 'White dots on black background'),
    ]
    
    dot_spacing = forms.IntegerField(
        min_value=10,
        max_value=50,
        initial=20,
        help_text="Distance between dots (10-50 pixels)",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '1'
        })
    )
    
    min_dot_radius = forms.IntegerField(
        min_value=1,
        max_value=5,
        initial=1,
        help_text="Minimum dot size (1-5 pixels)",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '1'
        })
    )
    
    max_dot_radius = forms.IntegerField(
        min_value=5,
        max_value=20,
        initial=10,
        help_text="Maximum dot size (5-20 pixels)",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '1'
        })
    )
    
    color_mode = forms.ChoiceField(
        choices=COLOR_MODE_CHOICES,
        initial='black_on_white',
        help_text="Choose dot and background colors",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    def clean_image(self):
        """Validate uploaded image file with enhanced dimension and size checks."""
        image = self.cleaned_data.get('image')
        
        if not image:
            raise ValidationError("Please select an image file.")
        
        # Check file extension
        name, ext = os.path.splitext(image.name.lower())
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.ico', '.psd', '.svg']
        if ext not in allowed_extensions:
            raise ValidationError("Unsupported file format. Please upload a valid image file.")
        
        # Check MIME type - allow all image types
        if not image.content_type.startswith('image/'):
            raise ValidationError("Invalid file format. Please upload a valid image file.")
        
        # Check file size (limit to 10MB)
        if image.size > 10 * 1024 * 1024:
            raise ValidationError("File size must be less than 10MB. Please compress your image or choose a smaller file.")
        
        # Validate image dimensions for optimal processing
        try:
            from PIL import Image as PILImage
            image.seek(0)  # Reset file pointer
            
            with PILImage.open(image) as img:
                width, height = img.size
                
                # Check minimum dimensions
                if width < 10 or height < 10:
                    raise ValidationError("Image dimensions are too small. Minimum size is 10x10 pixels.")
                
                # No maximum dimension restrictions - support any pixel size
                # Note: Very large images may take longer to process
            
            image.seek(0)  # Reset file pointer for later use
            
        except ValidationError:
            raise  # Re-raise validation errors
        except Exception as e:
            raise ValidationError("Unable to process the uploaded image. Please ensure it's a valid image file.")
        
        return image
    
    def clean(self):
        """Validate parameter relationships."""
        cleaned_data = super().clean()
        min_radius = cleaned_data.get('min_dot_radius')
        max_radius = cleaned_data.get('max_dot_radius')
        
        if min_radius and max_radius and min_radius >= max_radius:
            raise ValidationError("Minimum dot radius must be less than maximum dot radius.")
        
        return cleaned_data
    
    def clean_dot_spacing(self):
        """Validate dot spacing parameter."""
        dot_spacing = self.cleaned_data.get('dot_spacing')
        
        if dot_spacing is not None:
            if dot_spacing < 10 or dot_spacing > 50:
                raise ValidationError("Dot spacing must be between 10 and 50 pixels.")
        
        return dot_spacing
    
    def clean_min_dot_radius(self):
        """Validate minimum dot radius parameter."""
        min_radius = self.cleaned_data.get('min_dot_radius')
        
        if min_radius is not None:
            if min_radius < 1 or min_radius > 5:
                raise ValidationError("Minimum dot radius must be between 1 and 5 pixels.")
        
        return min_radius
    
    def clean_max_dot_radius(self):
        """Validate maximum dot radius parameter."""
        max_radius = self.cleaned_data.get('max_dot_radius')
        
        if max_radius is not None:
            if max_radius < 5 or max_radius > 20:
                raise ValidationError("Maximum dot radius must be between 5 and 20 pixels.")
        
        return max_radius