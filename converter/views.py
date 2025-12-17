"""
Views for the image dotting converter application.
"""
import os
import uuid
import logging
import tempfile
import shutil
from django.shortcuts import render, redirect
from django.views import View
from django.http import HttpResponse, Http404, FileResponse, JsonResponse
from django.contrib import messages
from django.conf import settings
from django.core.files.storage import default_storage
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from .forms import CombinedUploadForm
from .image_processor import ImageProcessor, ProcessingParameters, PencilSketchProcessor

# Set up logging
logger = logging.getLogger(__name__)


class UploadView(View):
    """View for handling image upload and parameter configuration."""
    
    def get(self, request):
        """Display the upload form."""
        # Clean up any previous session data
        self._cleanup_session_data(request)
        
        form = CombinedUploadForm()
        return render(request, 'converter/upload.html', {'form': form})
    
    def post(self, request):
        """Handle form submission and redirect to processing."""
        form = CombinedUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            try:
                # Clean up any previous session data first
                self._cleanup_session_data(request)
                
                # Save uploaded image to input directory
                uploaded_file = form.cleaned_data['image']
                
                # Reset file pointer to beginning (form validation may have moved it)
                uploaded_file.seek(0)
                
                # Generate unique filename to avoid conflicts
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                unique_filename = f"{uuid.uuid4()}{file_extension}"
                input_path = os.path.join('input', unique_filename)
                
                # Ensure input directory exists
                input_dir = os.path.join(settings.MEDIA_ROOT, 'input')
                os.makedirs(input_dir, exist_ok=True)
                
                # Save file using Django's default storage
                saved_path = default_storage.save(input_path, uploaded_file)
                
                # Log successful upload
                logger.info(f"File uploaded successfully: {saved_path}")
                
                # Store processing parameters and file path in session
                request.session['uploaded_file_path'] = saved_path
                request.session['conversion_type'] = form.cleaned_data['conversion_type']
                request.session['processing_params'] = {
                    'dot_spacing': form.cleaned_data['dot_spacing'],
                    'min_dot_radius': form.cleaned_data['min_dot_radius'],
                    'max_dot_radius': form.cleaned_data['max_dot_radius'],
                    'color_mode': form.cleaned_data['color_mode'],
                }
                
                # Add progress feedback
                messages.info(request, "Image uploaded successfully. Processing...")
                
                # Redirect to processing view
                return redirect('converter:process')
                
            except Exception as e:
                logger.error(f"Error uploading file: {str(e)}")
                messages.error(request, f"Error uploading file: {str(e)}")
                
        return render(request, 'converter/upload.html', {'form': form})
    
    def _validate_uploaded_file(self, uploaded_file):
        """Validate uploaded file format, size, and dimensions."""
        try:
            # Check file extension
            name, ext = os.path.splitext(uploaded_file.name.lower())
            if ext not in ['.jpg', '.jpeg', '.png']:
                return False
            
            # Check MIME type
            if uploaded_file.content_type not in ['image/png', 'image/jpeg']:
                return False
            
            # Check file size (10MB limit)
            if uploaded_file.size > 10 * 1024 * 1024:
                return False
            
            # Validate image dimensions for processing optimization
            try:
                from PIL import Image
                uploaded_file.seek(0)  # Reset file pointer
                with Image.open(uploaded_file) as img:
                    width, height = img.size
                    
                    # Check minimum dimensions
                    if width < 10 or height < 10:
                        return False
                    
                    # No maximum dimension restrictions - support any pixel size
                    # Note: Very large images may take longer to process
                
                uploaded_file.seek(0)  # Reset file pointer for later use
                return True
                
            except Exception as e:
                logger.warning(f"Error validating image dimensions: {str(e)}")
                return False
            
        except Exception:
            return False
    
    def _cleanup_session_data(self, request):
        """Clean up previous session data and associated files."""
        try:
            # Get previous file paths
            old_input_path = request.session.get('uploaded_file_path')
            old_output_path = request.session.get('output_file_path')
            
            # Remove old files if they exist
            if old_input_path:
                old_input_full_path = os.path.join(settings.MEDIA_ROOT, old_input_path)
                if os.path.exists(old_input_full_path):
                    os.remove(old_input_full_path)
                    logger.info(f"Cleaned up old input file: {old_input_path}")
            
            if old_output_path:
                old_output_full_path = os.path.join(settings.MEDIA_ROOT, old_output_path)
                if os.path.exists(old_output_full_path):
                    os.remove(old_output_full_path)
                    logger.info(f"Cleaned up old output file: {old_output_path}")
            
            # Clear session data
            request.session.pop('uploaded_file_path', None)
            request.session.pop('output_file_path', None)
            request.session.pop('processing_params', None)
            request.session.pop('conversion_type', None)
            
        except Exception as e:
            logger.warning(f"Error during session cleanup: {str(e)}")


class ProcessView(View):
    """View for processing images and displaying results."""
    
    def get(self, request):
        """Process the uploaded image and display results."""
        # Check if we have required session data
        uploaded_file_path = request.session.get('uploaded_file_path')
        conversion_type = request.session.get('conversion_type', 'dotting')
        processing_params = request.session.get('processing_params')
        
        if not uploaded_file_path:
            messages.error(request, "No image uploaded. Please upload an image first.")
            return redirect('converter:upload')
        
        # For pencil sketch, we don't need processing params
        if conversion_type == 'dotting' and not processing_params:
            messages.error(request, "No processing parameters found. Please upload an image first.")
            return redirect('converter:upload')
        
        # Verify input file still exists
        input_full_path = os.path.join(settings.MEDIA_ROOT, uploaded_file_path)
        if not os.path.exists(input_full_path):
            messages.error(request, "Uploaded image file not found. Please upload again.")
            return redirect('converter:upload')
        
        try:
            # Add progress feedback based on conversion type
            if conversion_type == 'pencil_sketch':
                messages.info(request, "Converting to pencil sketch... This may take a moment.")
            else:
                messages.info(request, "Processing image... This may take a moment.")
            
            # Generate output filename with timestamp for uniqueness
            input_filename = os.path.basename(uploaded_file_path)
            name, ext = os.path.splitext(input_filename)
            timestamp = str(int(uuid.uuid4().int))[:8]
            
            if conversion_type == 'pencil_sketch':
                output_filename = f"sketch_{name}_{timestamp}{ext}"
                processor = PencilSketchProcessor()
            else:
                # Create processing parameters object with validation
                params = self._create_validated_parameters(processing_params)
                if not params:
                    messages.error(request, "Invalid processing parameters. Please try again.")
                    return redirect('converter:upload')
                
                output_filename = f"dotted_{name}_{timestamp}{ext}"
                processor = ImageProcessor(params)
            
            # Get full paths
            output_path = os.path.join('output', output_filename)
            output_full_path = os.path.join(settings.MEDIA_ROOT, output_path)
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_full_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Process the image with error handling
            success = self._process_image_safely(processor, input_full_path, output_full_path)
            
            if success:
                # Verify output file was created
                if not os.path.exists(output_full_path):
                    messages.error(request, "Image processing completed but output file was not created. Please try again.")
                    return redirect('converter:upload')
                
                # Store output path in session for download
                request.session['output_file_path'] = output_path
                
                # Log successful processing
                logger.info(f"Image processed successfully: {output_path}")
                
                # Add success message based on conversion type
                if conversion_type == 'pencil_sketch':
                    messages.success(request, "Image converted to pencil sketch successfully!")
                else:
                    messages.success(request, "Image converted to dots successfully!")
                
                # Prepare context for template
                context = {
                    'original_image_url': settings.MEDIA_URL + uploaded_file_path,
                    'converted_image_url': settings.MEDIA_URL + output_path,
                    'download_filename': output_filename,
                    'conversion_type': conversion_type,
                    'processing_params': processing_params,
                }
                
                return render(request, 'converter/results.html', context)
            else:
                messages.error(request, "Error processing image. This may be due to a corrupted file or unsupported format. Please try again with a different image file.")
                return redirect('converter:upload')
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            messages.error(request, f"Unexpected error during processing: {str(e)}")
            return redirect('converter:upload')
    
    def _create_validated_parameters(self, processing_params):
        """Create and validate processing parameters."""
        try:
            params = ProcessingParameters(
                dot_spacing=processing_params['dot_spacing'],
                min_dot_radius=processing_params['min_dot_radius'],
                max_dot_radius=processing_params['max_dot_radius'],
                color_mode=processing_params['color_mode']
            )
            
            # Validate parameters
            if not params.is_valid():
                logger.error(f"Invalid parameters: {params.validate()}")
                return None
            
            return params
        except Exception as e:
            logger.error(f"Error creating parameters: {str(e)}")
            return None
    
    def _process_image_safely(self, processor, input_path, output_path):
        """Process image with proper error handling and direct output."""
        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Process the image directly to final location
            # No temporary file needed - let the processor handle it
            success = processor.process_image(input_path, output_path)
            
            if success and os.path.exists(output_path):
                # Verify the output file has content
                file_size = os.path.getsize(output_path)
                if file_size > 0:
                    logger.info(f"Successfully processed image: {file_size} bytes")
                    return True
                else:
                    logger.error(f"Output file is empty: {output_path}")
                    return False
            else:
                logger.error(f"Processing failed or output file not created: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error during image processing: {str(e)}")
            return False


class DownloadView(View):
    """View for serving generated dotted images."""
    
    def get(self, request, filename):
        """Serve the generated image file."""
        try:
            # Security check: ensure filename doesn't contain path traversal
            if '..' in filename or '/' in filename or '\\' in filename:
                logger.warning(f"Potential path traversal attempt: {filename}")
                raise Http404("Invalid filename")
            
            # Construct the file path in the output directory
            file_path = os.path.join('output', filename)
            full_path = os.path.join(settings.MEDIA_ROOT, file_path)
            
            # Normalize path to prevent traversal
            full_path = os.path.normpath(full_path)
            expected_dir = os.path.normpath(os.path.join(settings.MEDIA_ROOT, 'output'))
            
            # Ensure the file is within the expected directory
            if not full_path.startswith(expected_dir):
                logger.warning(f"File access outside output directory attempted: {full_path}")
                raise Http404("File not found")
            
            # Check if file exists
            if not os.path.exists(full_path):
                logger.info(f"Requested file not found: {full_path}")
                raise Http404("File not found")
            
            # Determine content type based on file extension
            file_ext = os.path.splitext(filename)[1].lower()
            content_type = 'image/jpeg'
            if file_ext == '.png':
                content_type = 'image/png'
            
            # Log download attempt
            logger.info(f"Serving file download: {filename}")
            
            # Serve the file
            try:
                file_handle = open(full_path, 'rb')
                response = FileResponse(
                    file_handle,
                    content_type=content_type,
                    as_attachment=True,
                    filename=filename
                )
                
                # Add headers for better download experience
                response['Content-Length'] = os.path.getsize(full_path)
                response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                response['Pragma'] = 'no-cache'
                response['Expires'] = '0'
                
                return response
                
            except IOError as e:
                logger.error(f"Error reading file {full_path}: {str(e)}")
                raise Http404("Error reading file")
            
        except Http404:
            raise
        except Exception as e:
            logger.error(f"Error downloading file {filename}: {str(e)}")
            messages.error(request, f"Error downloading file: {str(e)}")
            return redirect('converter:upload')


class CleanupView(View):
    """View for cleaning up temporary files (for maintenance)."""
    
    @method_decorator(csrf_exempt)
    def post(self, request):
        """Clean up old temporary files."""
        try:
            cleaned_count = self._cleanup_old_files()
            return JsonResponse({
                'success': True,
                'message': f'Cleaned up {cleaned_count} old files'
            })
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            return JsonResponse({
                'success': False,
                'message': f'Error during cleanup: {str(e)}'
            })
    
    def _cleanup_old_files(self):
        """Clean up files older than 1 hour."""
        import time
        
        cleaned_count = 0
        current_time = time.time()
        one_hour_ago = current_time - 3600  # 1 hour in seconds
        
        # Clean input directory
        input_dir = os.path.join(settings.MEDIA_ROOT, 'input')
        if os.path.exists(input_dir):
            for filename in os.listdir(input_dir):
                file_path = os.path.join(input_dir, filename)
                if os.path.isfile(file_path):
                    file_mtime = os.path.getmtime(file_path)
                    if file_mtime < one_hour_ago:
                        try:
                            os.remove(file_path)
                            cleaned_count += 1
                            logger.info(f"Cleaned up old input file: {filename}")
                        except Exception as e:
                            logger.warning(f"Could not remove {file_path}: {str(e)}")
        
        # Clean output directory
        output_dir = os.path.join(settings.MEDIA_ROOT, 'output')
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                if os.path.isfile(file_path):
                    file_mtime = os.path.getmtime(file_path)
                    if file_mtime < one_hour_ago:
                        try:
                            os.remove(file_path)
                            cleaned_count += 1
                            logger.info(f"Cleaned up old output file: {filename}")
                        except Exception as e:
                            logger.warning(f"Could not remove {file_path}: {str(e)}")
        
        return cleaned_count