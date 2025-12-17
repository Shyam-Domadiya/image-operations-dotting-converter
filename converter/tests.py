from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.messages import get_messages
from unittest.mock import patch, MagicMock
import os
import tempfile
from PIL import Image
import io
from converter.views import ProcessView


class TemplateRenderingTestCase(TestCase):
    """Unit tests for template rendering functionality."""
    
    def setUp(self):
        """Set up test client and test data."""
        self.client = Client()
        
        # Create a test image in memory
        self.test_image = self._create_test_image()
        
    def _create_test_image(self):
        """Create a simple test image for upload testing."""
        # Create a simple 100x100 RGB image
        image = Image.new('RGB', (100, 100), color='red')
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        return SimpleUploadedFile(
            name='test_image.jpg',
            content=buffer.getvalue(),
            content_type='image/jpeg'
        )
    
    def test_upload_page_displays_file_upload_form(self):
        """Test upload page displays file upload form."""
        response = self.client.get(reverse('converter:upload'))
        
        # Check response status
        self.assertEqual(response.status_code, 200)
        
        # Check that the form is present
        self.assertContains(response, '<form method="post" enctype="multipart/form-data">')
        
        # Check for file input field
        self.assertContains(response, 'type="file"')
        self.assertContains(response, 'name="image"')
        
        # Check for parameter input fields
        self.assertContains(response, 'name="dot_spacing"')
        self.assertContains(response, 'name="min_dot_radius"')
        self.assertContains(response, 'name="max_dot_radius"')
        self.assertContains(response, 'name="color_mode"')
        
        # Check for submit button
        self.assertContains(response, 'type="submit"')
        self.assertContains(response, 'Convert to Dots')
        
        # Check for instructions
        self.assertContains(response, 'Instructions:')
        self.assertContains(response, 'Image Upload:')
        self.assertContains(response, 'Dot Spacing:')
        
        # Check for help text
        self.assertContains(response, 'Upload a PNG or JPG image file')
        self.assertContains(response, 'Distance between dots')
        
    def test_upload_page_shows_parameter_options(self):
        """Test upload page shows parameter configuration options."""
        response = self.client.get(reverse('converter:upload'))
        
        # Check for dot spacing field with proper attributes
        self.assertContains(response, 'Dot spacing')
        self.assertContains(response, '10-50 pixels')
        
        # Check for dot radius fields
        self.assertContains(response, 'Minimum dot size')
        self.assertContains(response, '1-5 pixels')
        self.assertContains(response, 'Maximum dot size')
        self.assertContains(response, '5-20 pixels')
        
        # Check for color mode options
        self.assertContains(response, 'Color mode')
        self.assertContains(response, 'Black dots on white background')
        self.assertContains(response, 'White dots on black background')
        
        # Check that form fields have proper CSS classes
        self.assertContains(response, 'class="form-control"')
        
    @patch('converter.views.ImageProcessor')
    @patch('converter.views.default_storage')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('shutil.move')
    def test_results_page_displays_both_images_and_download_button(self, mock_move, mock_makedirs, mock_exists, mock_storage, mock_processor):
        """Test results page displays both images and download button."""
        # Mock the storage save method
        mock_storage.save.return_value = 'input/test_image.jpg'
        
        # Mock file existence - return True for input file, False initially for temp, True after processing
        def mock_exists_side_effect(path):
            if 'input' in path:
                return True
            elif '.tmp' in path:
                return False  # Temp file doesn't exist initially
            elif 'output' in path:
                return True  # Output file exists after processing
            return True
        
        mock_exists.side_effect = mock_exists_side_effect
        
        # Mock the image processor
        mock_processor_instance = MagicMock()
        mock_processor_instance.process_image.return_value = True
        mock_processor.return_value = mock_processor_instance
        
        # First, upload an image to set up session data
        upload_data = {
            'image': self.test_image,
            'dot_spacing': 20,
            'min_dot_radius': 1,
            'max_dot_radius': 10,
            'color_mode': 'black_on_white'
        }
        
        upload_response = self.client.post(reverse('converter:upload'), upload_data)
        self.assertEqual(upload_response.status_code, 302)  # Should redirect to process
        
        # Now test the results page
        with patch('converter.views.settings') as mock_settings:
            mock_settings.MEDIA_URL = '/media/'
            mock_settings.MEDIA_ROOT = '/tmp/media'
            
            # Mock the ProcessView._process_image_safely method to return True
            with patch.object(ProcessView, '_process_image_safely', return_value=True):
                response = self.client.get(reverse('converter:process'))
                
                # Check response status
                self.assertEqual(response.status_code, 200)
                
                # Check that both images are displayed
                self.assertContains(response, 'Original Image')
                self.assertContains(response, 'Dotted Version')
                
                # Check for image tags
                self.assertContains(response, '<img src=')
                
                # Check for download button
                self.assertContains(response, 'Download Dotted Image')
            self.assertContains(response, 'btn btn-success')
            
            # Check for "Convert Another Image" link
            self.assertContains(response, 'Convert Another Image')
            
            # Check for parameters summary
            self.assertContains(response, 'Processing Parameters Used:')
            self.assertContains(response, 'Dot Spacing:')
            self.assertContains(response, 'Minimum Dot Radius:')
            self.assertContains(response, 'Maximum Dot Radius:')
            self.assertContains(response, 'Color Mode:')
            
            # Check parameter values are displayed
            self.assertContains(response, '20 pixels')  # dot_spacing
            self.assertContains(response, '1 pixels')   # min_dot_radius
            self.assertContains(response, '10 pixels')  # max_dot_radius
            self.assertContains(response, 'Black dots on white background')
    
    def test_upload_page_template_structure(self):
        """Test upload page has proper template structure."""
        response = self.client.get(reverse('converter:upload'))
        
        # Check that it extends base template
        self.assertContains(response, 'Image Dotting Converter')
        
        # Check for proper CSS classes and styling
        self.assertContains(response, 'form-group')
        self.assertContains(response, 'help-text')
        
        # Check for CSRF token
        self.assertContains(response, 'csrfmiddlewaretoken')
        
        # Check for proper form encoding
        self.assertContains(response, 'enctype="multipart/form-data"')
    
    def test_upload_page_error_display(self):
        """Test upload page displays form errors properly."""
        # Submit form with invalid data
        invalid_data = {
            'dot_spacing': 100,  # Invalid: too high
            'min_dot_radius': 10,  # Invalid: higher than max
            'max_dot_radius': 5,   # Invalid: lower than min
        }
        
        response = self.client.post(reverse('converter:upload'), invalid_data)
        
        # Should stay on upload page with errors
        self.assertEqual(response.status_code, 200)
        
        # Check for error display elements
        self.assertContains(response, 'errorlist')
        
    def test_results_page_without_session_data_redirects(self):
        """Test results page redirects when no session data exists."""
        # Try to access results page without uploading first
        response = self.client.get(reverse('converter:process'))
        
        # Should redirect to upload page
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse('converter:upload'))
    
    def test_base_template_structure(self):
        """Test base template provides proper structure."""
        response = self.client.get(reverse('converter:upload'))
        
        # Check for proper HTML structure
        self.assertContains(response, '<!DOCTYPE html>')
        self.assertContains(response, '<html lang="en">')
        self.assertContains(response, '<meta charset="UTF-8">')
        self.assertContains(response, '<meta name="viewport"')
        
        # Check for CSS link
        self.assertContains(response, '<link rel="stylesheet"')
        self.assertContains(response, 'css/style.css')
        
        # Check for container structure
        self.assertContains(response, 'class="container"')
        
        # Check for title
        self.assertContains(response, '<title>')
        self.assertContains(response, 'Image Dotting Converter')
    
    def test_template_responsive_design(self):
        """Test templates include responsive design elements."""
        response = self.client.get(reverse('converter:upload'))
        
        # Check for viewport meta tag
        self.assertContains(response, 'width=device-width, initial-scale=1.0')
        
        # Check for external CSS file link (responsive styles are in external CSS)
        self.assertContains(response, 'css/style.css')
        
        # Check for responsive form classes
        self.assertContains(response, 'form-control')
        
    def test_template_accessibility_features(self):
        """Test templates include basic accessibility features."""
        response = self.client.get(reverse('converter:upload'))
        
        # Check for proper label associations
        self.assertContains(response, '<label for=')
        
        # Check for alt text structure (in results template test)
        # This will be tested when we have actual image display
        
        # Check for semantic HTML structure
        self.assertContains(response, '<h1>')
        self.assertContains(response, '<h3>')


class TemplateContentTestCase(TestCase):
    """Test template content and messaging."""
    
    def setUp(self):
        """Set up test client."""
        self.client = Client()
    
    def test_upload_page_instructions_content(self):
        """Test upload page contains proper instructions."""
        response = self.client.get(reverse('converter:upload'))
        
        # Check for comprehensive instructions
        self.assertContains(response, 'Upload a PNG or JPG image and configure')
        self.assertContains(response, 'pointillism effect')
        
        # Check for specific instruction items
        self.assertContains(response, 'max 10MB')
        self.assertContains(response, 'smaller = more dots')
        self.assertContains(response, 'size range of dots')
        
    def test_error_message_display_structure(self):
        """Test error messages are displayed with proper structure."""
        # Test with invalid file upload
        invalid_file = SimpleUploadedFile(
            name='test.txt',
            content=b'not an image',
            content_type='text/plain'
        )
        
        response = self.client.post(reverse('converter:upload'), {
            'image': invalid_file,
            'dot_spacing': 20,
            'min_dot_radius': 1,
            'max_dot_radius': 10,
            'color_mode': 'black_on_white'
        })
        
        # Should stay on upload page
        self.assertEqual(response.status_code, 200)
        
        # Check for error display
        self.assertContains(response, 'errorlist')
        
    def test_success_message_display_in_results(self):
        """Test success message is displayed in results page."""
        # This will be tested when we have a working results page
        # For now, we test the template structure exists
        pass


class ColorModeFunctionalityTestCase(TestCase):
    """Unit tests for color mode functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.client = Client()
        
        # Create a test image in memory
        self.test_image = self._create_test_image()
    
    def _create_test_image(self):
        """Create a simple test image for testing."""
        # Create a simple 100x100 RGB image
        image = Image.new('RGB', (100, 100), color='red')
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        return SimpleUploadedFile(
            name='test_image.jpg',
            content=buffer.getvalue(),
            content_type='image/jpeg'
        )
    
    def test_black_dots_on_white_background_mode(self):
        """
        Test black dots on white background mode.
        _Requirements: 2.5_
        """
        from converter.image_processor import ImageProcessor, ProcessingParameters
        import numpy as np
        
        # Create parameters for black on white mode
        parameters = ProcessingParameters(
            dot_spacing=20,
            min_dot_radius=2,
            max_dot_radius=8,
            color_mode='black_on_white'
        )
        
        processor = ImageProcessor(parameters)
        
        # Create a test canvas
        canvas = processor.create_fresh_canvas((100, 100))
        
        # Verify canvas is white background
        self.assertEqual(canvas.shape, (100, 100, 3))
        self.assertEqual(canvas.dtype, np.uint8)
        
        # Check that all pixels are white (255, 255, 255)
        expected_white = np.array([255, 255, 255], dtype=np.uint8)
        unique_colors = np.unique(canvas.reshape(-1, 3), axis=0)
        self.assertEqual(len(unique_colors), 1)
        np.testing.assert_array_equal(unique_colors[0], expected_white)
        
        # Test dot rendering with black dots
        # Create a simple grayscale image (dark area should produce large dots)
        grayscale_image = np.zeros((100, 100), dtype=np.uint8)  # All black (dark)
        
        # Render dots on canvas
        result = processor.render_dots(canvas.copy(), grayscale_image)
        
        # Verify result has both white background and black dots
        unique_colors_result = np.unique(result.reshape(-1, 3), axis=0)
        
        # Should have at least white background
        white_present = any(np.array_equal(color, [255, 255, 255]) for color in unique_colors_result)
        self.assertTrue(white_present, "White background should be present")
        
        # Should have black dots (since grayscale is all black, should produce large dots)
        black_present = any(np.array_equal(color, [0, 0, 0]) for color in unique_colors_result)
        self.assertTrue(black_present, "Black dots should be present for dark areas")
    
    def test_white_dots_on_black_background_mode(self):
        """
        Test white dots on black background mode.
        _Requirements: 2.5_
        """
        from converter.image_processor import ImageProcessor, ProcessingParameters
        import numpy as np
        
        # Create parameters for white on black mode
        parameters = ProcessingParameters(
            dot_spacing=20,
            min_dot_radius=2,
            max_dot_radius=8,
            color_mode='white_on_black'
        )
        
        processor = ImageProcessor(parameters)
        
        # Create a test canvas
        canvas = processor.create_fresh_canvas((100, 100))
        
        # Verify canvas is black background
        self.assertEqual(canvas.shape, (100, 100, 3))
        self.assertEqual(canvas.dtype, np.uint8)
        
        # Check that all pixels are black (0, 0, 0)
        expected_black = np.array([0, 0, 0], dtype=np.uint8)
        unique_colors = np.unique(canvas.reshape(-1, 3), axis=0)
        self.assertEqual(len(unique_colors), 1)
        np.testing.assert_array_equal(unique_colors[0], expected_black)
        
        # Test dot rendering with white dots
        # Create a simple grayscale image (dark area should produce large dots)
        grayscale_image = np.zeros((100, 100), dtype=np.uint8)  # All black (dark)
        
        # Render dots on canvas
        result = processor.render_dots(canvas.copy(), grayscale_image)
        
        # Verify result has both black background and white dots
        unique_colors_result = np.unique(result.reshape(-1, 3), axis=0)
        
        # Should have at least black background
        black_present = any(np.array_equal(color, [0, 0, 0]) for color in unique_colors_result)
        self.assertTrue(black_present, "Black background should be present")
        
        # Should have white dots (since grayscale is all black, should produce large dots)
        white_present = any(np.array_equal(color, [255, 255, 255]) for color in unique_colors_result)
        self.assertTrue(white_present, "White dots should be present for dark areas")
    
    def test_color_mode_form_validation(self):
        """Test color mode form validation accepts valid modes."""
        from converter.forms import ProcessingParametersForm
        
        # Test black_on_white mode
        form_data_black = {
            'dot_spacing': 20,
            'min_dot_radius': 2,
            'max_dot_radius': 8,
            'color_mode': 'black_on_white'
        }
        
        form_black = ProcessingParametersForm(data=form_data_black)
        self.assertTrue(form_black.is_valid(), "Form should be valid for black_on_white mode")
        self.assertEqual(form_black.cleaned_data['color_mode'], 'black_on_white')
        
        # Test white_on_black mode
        form_data_white = {
            'dot_spacing': 20,
            'min_dot_radius': 2,
            'max_dot_radius': 8,
            'color_mode': 'white_on_black'
        }
        
        form_white = ProcessingParametersForm(data=form_data_white)
        self.assertTrue(form_white.is_valid(), "Form should be valid for white_on_black mode")
        self.assertEqual(form_white.cleaned_data['color_mode'], 'white_on_black')
    
    def test_color_mode_form_displays_options(self):
        """Test that color mode options are displayed in the form."""
        response = self.client.get(reverse('converter:upload'))
        
        # Check that both color mode options are present
        self.assertContains(response, 'Black dots on white background')
        self.assertContains(response, 'White dots on black background')
        
        # Check that the select field is present
        self.assertContains(response, 'name="color_mode"')
        self.assertContains(response, 'value="black_on_white"')
        self.assertContains(response, 'value="white_on_black"')
    
    def test_color_mode_parameter_validation(self):
        """Test color mode parameter validation in ProcessingParameters class."""
        from converter.models import ProcessingParameters
        
        # Test valid color modes
        valid_params_black = ProcessingParameters(color_mode='black_on_white')
        self.assertTrue(valid_params_black.is_valid())
        self.assertEqual(len(valid_params_black.validate()), 0)
        
        valid_params_white = ProcessingParameters(color_mode='white_on_black')
        self.assertTrue(valid_params_white.is_valid())
        self.assertEqual(len(valid_params_white.validate()), 0)
        
        # Test invalid color mode
        invalid_params = ProcessingParameters(color_mode='invalid_mode')
        self.assertFalse(invalid_params.is_valid())
        errors = invalid_params.validate()
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('Color mode' in error for error in errors))
    
    def test_color_mode_integration_with_upload_form(self):
        """Test color mode integration with the combined upload form."""
        # Test form submission with black_on_white mode
        upload_data_black = {
            'image': self.test_image,
            'dot_spacing': 20,
            'min_dot_radius': 2,
            'max_dot_radius': 8,
            'color_mode': 'black_on_white'
        }
        
        response_black = self.client.post(reverse('converter:upload'), upload_data_black)
        # Should redirect to process view (successful form submission)
        self.assertEqual(response_black.status_code, 302)
        
        # Reset test image for second test
        self.test_image.seek(0)
        
        # Test form submission with white_on_black mode
        upload_data_white = {
            'image': self.test_image,
            'dot_spacing': 20,
            'min_dot_radius': 2,
            'max_dot_radius': 8,
            'color_mode': 'white_on_black'
        }
        
        response_white = self.client.post(reverse('converter:upload'), upload_data_white)
        # Should redirect to process view (successful form submission)
        self.assertEqual(response_white.status_code, 302)
    
    def test_color_mode_dot_color_consistency(self):
        """Test that dot colors are consistent with the selected color mode."""
        from converter.image_processor import ImageProcessor, ProcessingParameters
        import numpy as np
        
        # Test black_on_white mode dot colors
        params_black = ProcessingParameters(
            dot_spacing=10,  # Small spacing for more dots
            min_dot_radius=3,
            max_dot_radius=5,
            color_mode='black_on_white'
        )
        
        processor_black = ImageProcessor(params_black)
        canvas_black = processor_black.create_fresh_canvas((50, 50))
        
        # Create grayscale image with varying brightness
        grayscale = np.full((50, 50), 100, dtype=np.uint8)  # Medium gray
        
        result_black = processor_black.render_dots(canvas_black, grayscale)
        
        # Analyze colors in result
        unique_colors_black = np.unique(result_black.reshape(-1, 3), axis=0)
        
        # Should contain white background
        white_present = any(np.array_equal(color, [255, 255, 255]) for color in unique_colors_black)
        self.assertTrue(white_present, "Black_on_white mode should have white background")
        
        # Should contain black dots
        black_present = any(np.array_equal(color, [0, 0, 0]) for color in unique_colors_black)
        self.assertTrue(black_present, "Black_on_white mode should have black dots")
        
        # Test white_on_black mode dot colors
        params_white = ProcessingParameters(
            dot_spacing=10,  # Small spacing for more dots
            min_dot_radius=3,
            max_dot_radius=5,
            color_mode='white_on_black'
        )
        
        processor_white = ImageProcessor(params_white)
        canvas_white = processor_white.create_fresh_canvas((50, 50))
        
        result_white = processor_white.render_dots(canvas_white, grayscale)
        
        # Analyze colors in result
        unique_colors_white = np.unique(result_white.reshape(-1, 3), axis=0)
        
        # Should contain black background
        black_bg_present = any(np.array_equal(color, [0, 0, 0]) for color in unique_colors_white)
        self.assertTrue(black_bg_present, "White_on_black mode should have black background")
        
        # Should contain white dots
        white_dots_present = any(np.array_equal(color, [255, 255, 255]) for color in unique_colors_white)
        self.assertTrue(white_dots_present, "White_on_black mode should have white dots")
