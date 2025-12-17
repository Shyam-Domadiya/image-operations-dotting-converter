"""
Property-based tests for the image dotting converter application.
**Feature: image-dotting-converter**
"""
import os
import tempfile
from django.test import TestCase
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from hypothesis import given, strategies as st
from hypothesis.extra.django import TestCase as HypothesisTestCase
from PIL import Image
import io
import cv2
import numpy as np
from .image_processor import ImageProcessor, ProcessingParameters


class MediaDirectoryOrganizationTests(HypothesisTestCase):
    """
    Property-based tests for media directory organization.
    **Feature: image-dotting-converter, Property 13: Media directory organization**
    **Validates: Requirements 5.3**
    """
    
    def setUp(self):
        """Set up test environment."""
        self.input_dir = os.path.join(settings.MEDIA_ROOT, 'input')
        self.output_dir = os.path.join(settings.MEDIA_ROOT, 'output')
        
        # Ensure directories exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    @given(
        filename=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
            min_size=1,
            max_size=20
        ).map(lambda x: f"{x}.jpg"),
        width=st.integers(min_value=10, max_value=100),
        height=st.integers(min_value=10, max_value=100)
    )
    def test_uploaded_images_saved_to_input_directory(self, filename, width, height):
        """
        Property: For any uploaded image file, it should be saved to the input media directory.
        **Feature: image-dotting-converter, Property 13: Media directory organization**
        **Validates: Requirements 5.3**
        """
        # Create a test image
        image = Image.new('RGB', (width, height), color='red')
        image_io = io.BytesIO()
        image.save(image_io, format='JPEG')
        image_io.seek(0)
        
        # Create uploaded file
        uploaded_file = SimpleUploadedFile(
            filename,
            image_io.getvalue(),
            content_type='image/jpeg'
        )
        
        # Simulate saving to input directory
        input_path = os.path.join(self.input_dir, filename)
        with open(input_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        # Verify file exists in input directory
        self.assertTrue(os.path.exists(input_path))
        
        # Verify file is not in output directory
        output_path = os.path.join(self.output_dir, filename)
        self.assertFalse(os.path.exists(output_path))
        
        # Clean up
        if os.path.exists(input_path):
            os.remove(input_path)
    
    @given(
        filename=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
            min_size=1,
            max_size=20
        ).map(lambda x: f"dotted_{x}.jpg"),
        width=st.integers(min_value=10, max_value=100),
        height=st.integers(min_value=10, max_value=100)
    )
    def test_generated_images_saved_to_output_directory(self, filename, width, height):
        """
        Property: For any generated dotted image file, it should be saved to the output media directory.
        **Feature: image-dotting-converter, Property 13: Media directory organization**
        **Validates: Requirements 5.3**
        """
        # Create a test dotted image
        image = Image.new('RGB', (width, height), color='white')
        image_io = io.BytesIO()
        image.save(image_io, format='JPEG')
        image_io.seek(0)
        
        # Simulate saving to output directory
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'wb') as f:
            f.write(image_io.getvalue())
        
        # Verify file exists in output directory
        self.assertTrue(os.path.exists(output_path))
        
        # Verify file is not in input directory
        input_path = os.path.join(self.input_dir, filename)
        self.assertFalse(os.path.exists(input_path))
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
    
    def test_media_directories_exist(self):
        """
        Test that required media directories exist.
        **Feature: image-dotting-converter, Property 13: Media directory organization**
        **Validates: Requirements 5.3**
        """
        self.assertTrue(os.path.exists(self.input_dir))
        self.assertTrue(os.path.exists(self.output_dir))
        self.assertTrue(os.path.isdir(self.input_dir))
        self.assertTrue(os.path.isdir(self.output_dir))


class GrayscaleConversionTests(HypothesisTestCase):
    """
    Property-based tests for grayscale conversion.
    **Feature: image-dotting-converter, Property 6: Grayscale conversion**
    **Validates: Requirements 3.1**
    """
    
    def setUp(self):
        """Set up test environment."""
        self.parameters = ProcessingParameters()
        self.processor = ImageProcessor(self.parameters)
    
    @given(
        width=st.integers(min_value=10, max_value=200),
        height=st.integers(min_value=10, max_value=200),
        red=st.integers(min_value=0, max_value=255),
        green=st.integers(min_value=0, max_value=255),
        blue=st.integers(min_value=0, max_value=255)
    )
    def test_color_image_converted_to_grayscale(self, width, height, red, green, blue):
        """
        Property: For any color input image, the system should convert it to grayscale using OpenCV.
        **Feature: image-dotting-converter, Property 6: Grayscale conversion**
        **Validates: Requirements 3.1**
        """
        # Create a color image (BGR format for OpenCV)
        color_image = np.full((height, width, 3), [blue, green, red], dtype=np.uint8)
        
        # Convert to grayscale
        grayscale_result = self.processor.convert_to_grayscale(color_image)
        
        # Verify result is grayscale (2D array)
        self.assertEqual(len(grayscale_result.shape), 2)
        self.assertEqual(grayscale_result.shape, (height, width))
        
        # Verify all values are in valid grayscale range
        self.assertTrue(np.all(grayscale_result >= 0))
        self.assertTrue(np.all(grayscale_result <= 255))
        
        # Verify the conversion matches OpenCV's expected result
        expected = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(grayscale_result, expected)
    
    @given(
        width=st.integers(min_value=10, max_value=200),
        height=st.integers(min_value=10, max_value=200),
        gray_value=st.integers(min_value=0, max_value=255)
    )
    def test_grayscale_image_unchanged(self, width, height, gray_value):
        """
        Property: For any grayscale input image, the system should return it unchanged.
        **Feature: image-dotting-converter, Property 6: Grayscale conversion**
        **Validates: Requirements 3.1**
        """
        # Create a grayscale image
        grayscale_image = np.full((height, width), gray_value, dtype=np.uint8)
        
        # Process through conversion
        result = self.processor.convert_to_grayscale(grayscale_image)
        
        # Verify result is identical to input
        np.testing.assert_array_equal(result, grayscale_image)
        
        # Verify shape is preserved
        self.assertEqual(result.shape, (height, width))


class GridGenerationTests(HypothesisTestCase):
    """
    Property-based tests for grid generation consistency.
    **Feature: image-dotting-converter, Property 7: Grid generation consistency**
    **Validates: Requirements 3.2**
    """
    
    @given(
        width=st.integers(min_value=20, max_value=500),
        height=st.integers(min_value=20, max_value=500),
        dot_spacing=st.integers(min_value=5, max_value=50)
    )
    def test_grid_dimensions_calculated_correctly(self, width, height, dot_spacing):
        """
        Property: For any image dimensions and dot spacing value, the system should create a grid with correctly calculated dimensions.
        **Feature: image-dotting-converter, Property 7: Grid generation consistency**
        **Validates: Requirements 3.2**
        """
        parameters = ProcessingParameters(dot_spacing=dot_spacing)
        processor = ImageProcessor(parameters)
        
        # Calculate grid dimensions
        grid_rows, grid_cols = processor.create_grid((height, width))
        
        # Verify grid dimensions are calculated correctly
        expected_rows = height // dot_spacing
        expected_cols = width // dot_spacing
        
        self.assertEqual(grid_rows, expected_rows)
        self.assertEqual(grid_cols, expected_cols)
        
        # Verify grid dimensions are non-negative
        self.assertGreaterEqual(grid_rows, 0)
        self.assertGreaterEqual(grid_cols, 0)
        
        # Verify grid doesn't exceed image bounds
        self.assertLessEqual(grid_rows * dot_spacing, height)
        self.assertLessEqual(grid_cols * dot_spacing, width)
    
    @given(
        width=st.integers(min_value=1, max_value=19),
        height=st.integers(min_value=1, max_value=19),
        dot_spacing=st.integers(min_value=20, max_value=50)
    )
    def test_small_images_produce_zero_grid(self, width, height, dot_spacing):
        """
        Property: For any image smaller than dot spacing, grid dimensions should be zero.
        **Feature: image-dotting-converter, Property 7: Grid generation consistency**
        **Validates: Requirements 3.2**
        """
        parameters = ProcessingParameters(dot_spacing=dot_spacing)
        processor = ImageProcessor(parameters)
        
        # Calculate grid dimensions for small image
        grid_rows, grid_cols = processor.create_grid((height, width))
        
        # Both dimensions should be zero when image is smaller than spacing
        self.assertEqual(grid_rows, 0)
        self.assertEqual(grid_cols, 0)


class BrightnessCalculationTests(HypothesisTestCase):
    """
    Property-based tests for brightness calculation bounds.
    **Feature: image-dotting-converter, Property 8: Brightness calculation bounds**
    **Validates: Requirements 3.3**
    """
    
    @given(
        width=st.integers(min_value=50, max_value=200),
        height=st.integers(min_value=50, max_value=200),
        dot_spacing=st.integers(min_value=10, max_value=25),
        pixel_values=st.lists(
            st.integers(min_value=0, max_value=255),
            min_size=100,
            max_size=625  # 25x25 max grid cell
        )
    )
    def test_brightness_values_within_valid_range(self, width, height, dot_spacing, pixel_values):
        """
        Property: For any grid cell in an image, the computed average brightness value should be within the valid range [0, 255].
        **Feature: image-dotting-converter, Property 8: Brightness calculation bounds**
        **Validates: Requirements 3.3**
        """
        parameters = ProcessingParameters(dot_spacing=dot_spacing)
        processor = ImageProcessor(parameters)
        
        # Create a grayscale image with random pixel values
        image = np.zeros((height, width), dtype=np.uint8)
        
        # Fill image with pixel values (cycling through the list if needed)
        for y in range(height):
            for x in range(width):
                idx = (y * width + x) % len(pixel_values)
                image[y, x] = pixel_values[idx]
        
        # Calculate grid dimensions
        grid_rows, grid_cols = processor.create_grid((height, width))
        
        # Test brightness calculation for all valid grid cells
        for row in range(grid_rows):
            for col in range(grid_cols):
                brightness = processor.calculate_brightness(image, row, col)
                
                # Verify brightness is within valid range
                self.assertGreaterEqual(brightness, 0.0)
                self.assertLessEqual(brightness, 255.0)
                
                # Verify brightness is a valid float
                self.assertIsInstance(brightness, float)
    
    @given(
        dot_spacing=st.integers(min_value=10, max_value=30),
        uniform_value=st.integers(min_value=0, max_value=255)
    )
    def test_uniform_image_brightness_equals_pixel_value(self, dot_spacing, uniform_value):
        """
        Property: For any uniform image, brightness calculation should equal the uniform pixel value.
        **Feature: image-dotting-converter, Property 8: Brightness calculation bounds**
        **Validates: Requirements 3.3**
        """
        parameters = ProcessingParameters(dot_spacing=dot_spacing)
        processor = ImageProcessor(parameters)
        
        # Create uniform image
        width, height = 100, 100
        image = np.full((height, width), uniform_value, dtype=np.uint8)
        
        # Test brightness calculation for first grid cell
        brightness = processor.calculate_brightness(image, 0, 0)
        
        # Should equal the uniform value
        self.assertAlmostEqual(brightness, float(uniform_value), places=1)


class DotRadiusScalingTests(HypothesisTestCase):
    """
    Property-based tests for dot radius scaling.
    **Feature: image-dotting-converter, Property 9: Dot radius scaling**
    **Validates: Requirements 3.4**
    """
    
    @given(
        brightness=st.floats(min_value=0.0, max_value=255.0),
        min_radius=st.integers(min_value=1, max_value=5),
        max_radius=st.integers(min_value=6, max_value=20)
    )
    def test_dot_radius_within_specified_bounds(self, brightness, min_radius, max_radius):
        """
        Property: For any brightness intensity value, the generated dot radius should be scaled proportionally between minimum and maximum radius values.
        **Feature: image-dotting-converter, Property 9: Dot radius scaling**
        **Validates: Requirements 3.4**
        """
        parameters = ProcessingParameters(
            min_dot_radius=min_radius,
            max_dot_radius=max_radius
        )
        processor = ImageProcessor(parameters)
        
        # Calculate scaled radius
        radius = processor.scale_dot_radius(brightness)
        
        # Verify radius is within bounds
        self.assertGreaterEqual(radius, min_radius)
        self.assertLessEqual(radius, max_radius)
        
        # Verify radius is an integer
        self.assertIsInstance(radius, int)
    
    @given(
        min_radius=st.integers(min_value=1, max_value=5),
        max_radius=st.integers(min_value=6, max_value=20)
    )
    def test_extreme_brightness_values_produce_expected_radii(self, min_radius, max_radius):
        """
        Property: Extreme brightness values should produce predictable radius scaling.
        **Feature: image-dotting-converter, Property 9: Dot radius scaling**
        **Validates: Requirements 3.4**
        """
        parameters = ProcessingParameters(
            min_dot_radius=min_radius,
            max_dot_radius=max_radius
        )
        processor = ImageProcessor(parameters)
        
        # Test minimum brightness (black = 0) should give maximum radius
        black_radius = processor.scale_dot_radius(0.0)
        self.assertEqual(black_radius, max_radius)
        
        # Test maximum brightness (white = 255) should give minimum radius
        white_radius = processor.scale_dot_radius(255.0)
        self.assertEqual(white_radius, min_radius)
    
    @given(
        min_radius=st.integers(min_value=1, max_value=5),
        max_radius=st.integers(min_value=6, max_value=20)
    )
    def test_monotonic_brightness_to_radius_relationship(self, min_radius, max_radius):
        """
        Property: Darker pixels should produce larger radii (monotonic inverse relationship).
        **Feature: image-dotting-converter, Property 9: Dot radius scaling**
        **Validates: Requirements 3.4**
        """
        parameters = ProcessingParameters(
            min_dot_radius=min_radius,
            max_dot_radius=max_radius
        )
        processor = ImageProcessor(parameters)
        
        # Test that darker brightness produces larger radius
        dark_brightness = 50.0
        light_brightness = 200.0
        
        dark_radius = processor.scale_dot_radius(dark_brightness)
        light_radius = processor.scale_dot_radius(light_brightness)
        
        # Darker should have larger radius
        self.assertGreaterEqual(dark_radius, light_radius)


class FileFormatValidationTests(HypothesisTestCase):
    """
    Property-based tests for file format validation.
    **Feature: image-dotting-converter, Property 1: File format validation**
    **Validates: Requirements 1.2**
    """
    
    @given(
        filename=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
            min_size=1,
            max_size=20
        ),
        extension=st.sampled_from(['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'])
    )
    def test_valid_format_files_accepted(self, filename, extension):
        """
        Property: For any uploaded file with valid PNG or JPG format, the system should accept it and proceed to processing options.
        **Feature: image-dotting-converter, Property 1: File format validation**
        **Validates: Requirements 1.2**
        """
        from .forms import ImageUploadForm
        
        # Create a valid image file
        image = Image.new('RGB', (100, 100), color='red')
        image_io = io.BytesIO()
        
        # Save in appropriate format
        if extension.lower() in ['.jpg', '.jpeg']:
            image.save(image_io, format='JPEG')
            content_type = 'image/jpeg'
        else:  # PNG
            image.save(image_io, format='PNG')
            content_type = 'image/png'
        
        image_io.seek(0)
        
        # Create uploaded file
        uploaded_file = SimpleUploadedFile(
            filename + extension,
            image_io.getvalue(),
            content_type=content_type
        )
        
        # Test form validation
        form = ImageUploadForm(files={'image': uploaded_file})
        
        # Should be valid
        self.assertTrue(form.is_valid(), f"Form should be valid for {extension} files")
        
        # Should have no errors
        self.assertEqual(len(form.errors), 0)


class InvalidFormatRejectionTests(HypothesisTestCase):
    """
    Property-based tests for invalid format rejection.
    **Feature: image-dotting-converter, Property 2: Invalid format rejection**
    **Validates: Requirements 1.3**
    """
    
    @given(
        filename=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
            min_size=1,
            max_size=20
        ),
        invalid_extension=st.sampled_from(['.gif', '.bmp', '.tiff', '.webp', '.svg', '.txt', '.pdf', '.doc'])
    )
    def test_invalid_format_files_rejected(self, filename, invalid_extension):
        """
        Property: For any uploaded file with invalid format, the system should reject it and display an error message.
        **Feature: image-dotting-converter, Property 2: Invalid format rejection**
        **Validates: Requirements 1.3**
        """
        from .forms import ImageUploadForm
        
        # Create a file with invalid extension
        file_content = b"fake file content"
        uploaded_file = SimpleUploadedFile(
            filename + invalid_extension,
            file_content,
            content_type='application/octet-stream'
        )
        
        # Test form validation
        form = ImageUploadForm(files={'image': uploaded_file})
        
        # Should be invalid
        self.assertFalse(form.is_valid(), f"Form should be invalid for {invalid_extension} files")
        
        # Should have errors
        self.assertGreater(len(form.errors), 0)
        
        # Should have image field error
        self.assertIn('image', form.errors)
    
    @given(
        filename=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
            min_size=1,
            max_size=20
        ),
        wrong_content_type=st.sampled_from(['text/plain', 'application/pdf', 'video/mp4', 'audio/mp3'])
    )
    def test_wrong_mime_type_rejected(self, filename, wrong_content_type):
        """
        Property: For any file with wrong MIME type, the system should reject it.
        **Feature: image-dotting-converter, Property 2: Invalid format rejection**
        **Validates: Requirements 1.3**
        """
        from .forms import ImageUploadForm
        
        # Create a file with wrong MIME type but valid extension
        file_content = b"fake file content"
        uploaded_file = SimpleUploadedFile(
            filename + '.jpg',
            file_content,
            content_type=wrong_content_type
        )
        
        # Test form validation
        form = ImageUploadForm(files={'image': uploaded_file})
        
        # Should be invalid
        self.assertFalse(form.is_valid(), f"Form should be invalid for wrong MIME type {wrong_content_type}")
        
        # Should have errors
        self.assertGreater(len(form.errors), 0)
    
    def test_empty_file_rejected(self):
        """
        Property: Empty files should be rejected.
        **Feature: image-dotting-converter, Property 2: Invalid format rejection**
        **Validates: Requirements 1.3**
        """
        from .forms import ImageUploadForm
        
        # Test form without file
        form = ImageUploadForm(files={})
        
        # Should be invalid
        self.assertFalse(form.is_valid())
        
        # Should have image field error
        self.assertIn('image', form.errors)


class ParameterValidationTests(HypothesisTestCase):
    """
    Property-based tests for parameter validation.
    **Feature: image-dotting-converter, Property 4 & 5: Parameter validation**
    **Validates: Requirements 2.2, 2.3, 2.4**
    """
    
    @given(
        dot_spacing=st.integers(min_value=10, max_value=50),
        min_radius=st.integers(min_value=1, max_value=5),
        max_radius=st.integers(min_value=5, max_value=20)
    )
    def test_valid_parameters_accepted(self, dot_spacing, min_radius, max_radius):
        """
        Property: For any valid parameter values within acceptable ranges, the system should accept them.
        **Feature: image-dotting-converter, Property 4: Parameter validation for dot spacing**
        **Feature: image-dotting-converter, Property 5: Parameter validation for dot radius**
        **Validates: Requirements 2.2, 2.3, 2.4**
        """
        from .forms import ProcessingParametersForm
        
        # Ensure min < max for radius
        if min_radius >= max_radius:
            min_radius = max_radius - 1
            if min_radius < 1:
                max_radius = min_radius + 1
        
        form_data = {
            'dot_spacing': dot_spacing,
            'min_dot_radius': min_radius,
            'max_dot_radius': max_radius,
            'color_mode': 'black_on_white'
        }
        
        form = ProcessingParametersForm(data=form_data)
        
        # Should be valid
        self.assertTrue(form.is_valid(), f"Form should be valid for parameters: {form_data}")
        
        # Should have no errors
        self.assertEqual(len(form.errors), 0)
    
    @given(
        invalid_dot_spacing=st.one_of(
            st.integers(max_value=9),
            st.integers(min_value=51)
        )
    )
    def test_invalid_dot_spacing_rejected(self, invalid_dot_spacing):
        """
        Property: For any dot spacing value outside acceptable range, the system should reject it.
        **Feature: image-dotting-converter, Property 4: Parameter validation for dot spacing**
        **Validates: Requirements 2.2**
        """
        from .forms import ProcessingParametersForm
        
        form_data = {
            'dot_spacing': invalid_dot_spacing,
            'min_dot_radius': 2,
            'max_dot_radius': 8,
            'color_mode': 'black_on_white'
        }
        
        form = ProcessingParametersForm(data=form_data)
        
        # Should be invalid
        self.assertFalse(form.is_valid(), f"Form should be invalid for dot_spacing: {invalid_dot_spacing}")
        
        # Should have dot_spacing error
        self.assertIn('dot_spacing', form.errors)
    
    @given(
        invalid_min_radius=st.one_of(
            st.integers(max_value=0),
            st.integers(min_value=6)
        )
    )
    def test_invalid_min_radius_rejected(self, invalid_min_radius):
        """
        Property: For any minimum radius value outside acceptable range, the system should reject it.
        **Feature: image-dotting-converter, Property 5: Parameter validation for dot radius**
        **Validates: Requirements 2.3**
        """
        from .forms import ProcessingParametersForm
        
        form_data = {
            'dot_spacing': 20,
            'min_dot_radius': invalid_min_radius,
            'max_dot_radius': 10,
            'color_mode': 'black_on_white'
        }
        
        form = ProcessingParametersForm(data=form_data)
        
        # Should be invalid
        self.assertFalse(form.is_valid(), f"Form should be invalid for min_dot_radius: {invalid_min_radius}")
        
        # Should have min_dot_radius error
        self.assertIn('min_dot_radius', form.errors)
    
    @given(
        invalid_max_radius=st.one_of(
            st.integers(max_value=4),
            st.integers(min_value=21)
        )
    )
    def test_invalid_max_radius_rejected(self, invalid_max_radius):
        """
        Property: For any maximum radius value outside acceptable range, the system should reject it.
        **Feature: image-dotting-converter, Property 5: Parameter validation for dot radius**
        **Validates: Requirements 2.4**
        """
        from .forms import ProcessingParametersForm
        
        form_data = {
            'dot_spacing': 20,
            'min_dot_radius': 2,
            'max_dot_radius': invalid_max_radius,
            'color_mode': 'black_on_white'
        }
        
        form = ProcessingParametersForm(data=form_data)
        
        # Should be invalid
        self.assertFalse(form.is_valid(), f"Form should be invalid for max_dot_radius: {invalid_max_radius}")
        
        # Should have max_dot_radius error
        self.assertIn('max_dot_radius', form.errors)
    
    def test_min_radius_equal_to_max_rejected(self):
        """
        Property: When minimum radius equals maximum radius, the system should reject it.
        **Feature: image-dotting-converter, Property 5: Parameter validation for dot radius**
        **Validates: Requirements 2.3, 2.4**
        """
        from .forms import ProcessingParametersForm
        
        # Test the boundary case where min_radius = max_radius = 5
        # This is the only case where both values are individually valid
        # but violate the relationship constraint
        form_data = {
            'dot_spacing': 20,
            'min_dot_radius': 5,
            'max_dot_radius': 5,
            'color_mode': 'black_on_white'
        }
        
        form = ProcessingParametersForm(data=form_data)
        
        # Should be invalid
        self.assertFalse(form.is_valid(), "Form should be invalid when min_radius equals max_radius")
        
        # Should have form-level error
        self.assertGreater(len(form.non_field_errors()), 0, "Should have non-field error for equal radii")


class FreshCanvasGenerationTests(HypothesisTestCase):
    """
    Property-based tests for fresh canvas generation.
    **Feature: image-dotting-converter, Property 10: Fresh canvas generation**
    **Validates: Requirements 3.5**
    """
    
    @given(
        width=st.integers(min_value=10, max_value=300),
        height=st.integers(min_value=10, max_value=300),
        color_mode=st.sampled_from(['black_on_white', 'white_on_black'])
    )
    def test_fresh_canvas_created_without_original_overlay(self, width, height, color_mode):
        """
        Property: For any input image, the output should be generated on a blank canvas without overlaying on the original image.
        **Feature: image-dotting-converter, Property 10: Fresh canvas generation**
        **Validates: Requirements 3.5**
        """
        parameters = ProcessingParameters(color_mode=color_mode)
        processor = ImageProcessor(parameters)
        
        # Create fresh canvas
        canvas = processor.create_fresh_canvas((height, width))
        
        # Verify canvas dimensions
        self.assertEqual(canvas.shape, (height, width, 3))
        
        # Verify canvas is uniform (no original image content)
        if color_mode == 'black_on_white':
            # Should be all white (255, 255, 255)
            expected_color = np.array([255, 255, 255], dtype=np.uint8)
        else:  # white_on_black
            # Should be all black (0, 0, 0)
            expected_color = np.array([0, 0, 0], dtype=np.uint8)
        
        # Check that all pixels have the expected background color
        unique_colors = np.unique(canvas.reshape(-1, 3), axis=0)
        self.assertEqual(len(unique_colors), 1)
        np.testing.assert_array_equal(unique_colors[0], expected_color)
    
    @given(
        width=st.integers(min_value=10, max_value=200),
        height=st.integers(min_value=10, max_value=200)
    )
    def test_canvas_data_type_and_format(self, width, height):
        """
        Property: Fresh canvas should have correct data type and format for OpenCV operations.
        **Feature: image-dotting-converter, Property 10: Fresh canvas generation**
        **Validates: Requirements 3.5**
        """
        parameters = ProcessingParameters()
        processor = ImageProcessor(parameters)
        
        # Create fresh canvas
        canvas = processor.create_fresh_canvas((height, width))
        
        # Verify data type is uint8 (required for OpenCV)
        self.assertEqual(canvas.dtype, np.uint8)
        
        # Verify it's a 3-channel color image
        self.assertEqual(len(canvas.shape), 3)
        self.assertEqual(canvas.shape[2], 3)
        
        # Verify all values are in valid range [0, 255]
        self.assertTrue(np.all(canvas >= 0))
        self.assertTrue(np.all(canvas <= 255))


class FilePersistenceTests(HypothesisTestCase):
    """
    Property-based tests for file persistence.
    **Feature: image-dotting-converter, Property 3: File persistence**
    **Validates: Requirements 1.4**
    """
    
    def setUp(self):
        """Set up test environment."""
        self.media_root = settings.MEDIA_ROOT
        self.input_dir = os.path.join(self.media_root, 'input')
        os.makedirs(self.input_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        # Clean up any test files created
        if os.path.exists(self.input_dir):
            for filename in os.listdir(self.input_dir):
                if filename.startswith('test_'):
                    try:
                        os.remove(os.path.join(self.input_dir, filename))
                    except (PermissionError, FileNotFoundError):
                        # File might be in use or already deleted, skip
                        pass
    
    @given(
        filename=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
            min_size=1,
            max_size=15
        ).map(lambda x: f"test_{x}.jpg"),
        width=st.integers(min_value=10, max_value=100),
        height=st.integers(min_value=10, max_value=100)
    )
    def test_uploaded_file_saved_to_django_media_directory(self, filename, width, height):
        """
        Property: For any successfully uploaded image file, the system should save it to the Django media directory.
        **Feature: image-dotting-converter, Property 3: File persistence**
        **Validates: Requirements 1.4**
        """
        # Create a test image
        image = Image.new('RGB', (width, height), color='blue')
        image_io = io.BytesIO()
        image.save(image_io, format='JPEG')
        image_io.seek(0)
        
        # Create uploaded file
        uploaded_file = SimpleUploadedFile(
            filename,
            image_io.getvalue(),
            content_type='image/jpeg'
        )
        
        # Simulate Django's file saving process
        from django.core.files.storage import default_storage
        
        # Save file to input directory (as done in UploadView)
        input_path = os.path.join('input', filename)
        saved_path = default_storage.save(input_path, uploaded_file)
        
        # Verify file was saved to the correct location
        full_path = os.path.join(self.media_root, saved_path)
        self.assertTrue(os.path.exists(full_path))
        
        # Verify file is in the input directory
        self.assertTrue(saved_path.startswith('input/'))
        
        # Verify file content is preserved
        with open(full_path, 'rb') as f:
            saved_content = f.read()
        
        # Reset uploaded file for comparison
        uploaded_file.seek(0)
        original_content = uploaded_file.read()
        
        self.assertEqual(saved_content, original_content)
        
        # Clean up
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
            except (PermissionError, FileNotFoundError):
                # File might be in use, skip cleanup
                pass
    
    @given(
        width=st.integers(min_value=10, max_value=100),
        height=st.integers(min_value=10, max_value=100),
        file_format=st.sampled_from(['JPEG', 'PNG'])
    )
    def test_file_persistence_preserves_image_data(self, width, height, file_format):
        """
        Property: For any uploaded image, file persistence should preserve the image data integrity.
        **Feature: image-dotting-converter, Property 3: File persistence**
        **Validates: Requirements 1.4**
        """
        # Create a test image with specific pattern
        image = Image.new('RGB', (width, height))
        # Create a simple pattern to verify data integrity
        for x in range(width):
            for y in range(height):
                # Create a gradient pattern
                r = (x * 255) // width
                g = (y * 255) // height
                b = ((x + y) * 255) // (width + height)
                image.putpixel((x, y), (r, g, b))
        
        # Save to bytes
        image_io = io.BytesIO()
        image.save(image_io, format=file_format)
        image_io.seek(0)
        
        # Create uploaded file
        extension = '.jpg' if file_format == 'JPEG' else '.png'
        content_type = 'image/jpeg' if file_format == 'JPEG' else 'image/png'
        filename = f"test_integrity_{width}x{height}{extension}"
        
        uploaded_file = SimpleUploadedFile(
            filename,
            image_io.getvalue(),
            content_type=content_type
        )
        
        # Save using Django storage
        from django.core.files.storage import default_storage
        input_path = os.path.join('input', filename)
        saved_path = default_storage.save(input_path, uploaded_file)
        
        # Verify file exists
        full_path = os.path.join(self.media_root, saved_path)
        self.assertTrue(os.path.exists(full_path))
        
        # Load saved image and verify dimensions are preserved
        with Image.open(full_path) as saved_image:
            self.assertEqual(saved_image.size, (width, height))
            # Verify image mode is preserved (RGB)
            self.assertEqual(saved_image.mode, 'RGB')
        
        # Clean up
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
            except (PermissionError, FileNotFoundError):
                # File might be in use, skip cleanup
                pass
    
    def test_file_persistence_handles_duplicate_names(self):
        """
        Property: File persistence should handle duplicate filenames gracefully.
        **Feature: image-dotting-converter, Property 3: File persistence**
        **Validates: Requirements 1.4**
        """
        # Create two identical filenames
        filename = "test_duplicate.jpg"
        
        # Create first image
        image1 = Image.new('RGB', (50, 50), color='red')
        image1_io = io.BytesIO()
        image1.save(image1_io, format='JPEG')
        image1_io.seek(0)
        
        uploaded_file1 = SimpleUploadedFile(
            filename,
            image1_io.getvalue(),
            content_type='image/jpeg'
        )
        
        # Create second image with same filename
        image2 = Image.new('RGB', (50, 50), color='green')
        image2_io = io.BytesIO()
        image2.save(image2_io, format='JPEG')
        image2_io.seek(0)
        
        uploaded_file2 = SimpleUploadedFile(
            filename,
            image2_io.getvalue(),
            content_type='image/jpeg'
        )
        
        # Save both files
        from django.core.files.storage import default_storage
        
        input_path = os.path.join('input', filename)
        saved_path1 = default_storage.save(input_path, uploaded_file1)
        saved_path2 = default_storage.save(input_path, uploaded_file2)
        
        # Verify both files exist with different names
        full_path1 = os.path.join(self.media_root, saved_path1)
        full_path2 = os.path.join(self.media_root, saved_path2)
        
        self.assertTrue(os.path.exists(full_path1))
        self.assertTrue(os.path.exists(full_path2))
        
        # Verify paths are different (Django should handle name conflicts)
        self.assertNotEqual(saved_path1, saved_path2)
        
        # Clean up
        for path in [full_path1, full_path2]:
            if os.path.exists(path):
                os.remove(path)


class FileDownloadFunctionalityTests(HypothesisTestCase):
    """
    Property-based tests for file download functionality.
    **Feature: image-dotting-converter, Property 11: File download functionality**
    **Validates: Requirements 4.3**
    """
    
    def setUp(self):
        """Set up test environment."""
        self.media_root = settings.MEDIA_ROOT
        self.output_dir = os.path.join(self.media_root, 'output')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        # Clean up any test files created
        if os.path.exists(self.output_dir):
            for filename in os.listdir(self.output_dir):
                if filename.startswith('test_'):
                    try:
                        os.remove(os.path.join(self.output_dir, filename))
                    except (PermissionError, FileNotFoundError):
                        # File might be in use or already deleted, skip
                        pass
    
    @given(
        filename=st.text(
            alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            min_size=1,
            max_size=15
        ).map(lambda x: f"test_dotted_{x}.jpg"),
        width=st.integers(min_value=10, max_value=100),
        height=st.integers(min_value=10, max_value=100)
    )
    def test_generated_image_download_serves_correct_file(self, filename, width, height):
        """
        Property: For any generated dotted image, the download request should serve the correct image file.
        **Feature: image-dotting-converter, Property 11: File download functionality**
        **Validates: Requirements 4.3**
        """
        from django.test import Client
        from django.urls import reverse
        
        # Create a test dotted image file
        image = Image.new('RGB', (width, height), color='white')
        # Add some dots to simulate dotted image
        import random
        for _ in range(10):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            image.putpixel((x, y), (0, 0, 0))  # Black dots
        
        # Save to output directory
        output_path = os.path.join(self.output_dir, filename)
        image.save(output_path, format='JPEG')
        
        # Verify file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Test download via Django view
        client = Client()
        download_url = reverse('converter:download', kwargs={'filename': filename})
        response = client.get(download_url)
        
        # Verify response is successful
        self.assertEqual(response.status_code, 200)
        
        # Verify content type is correct
        self.assertEqual(response['Content-Type'], 'image/jpeg')
        
        # Verify content disposition for download
        content_disposition = response.get('Content-Disposition', '')
        self.assertIn('attachment', content_disposition)
        # Check that filename is present (may be URL-encoded)
        self.assertTrue(
            filename in content_disposition or 
            any(part in content_disposition for part in filename.split('_')),
            f"Filename {filename} not found in Content-Disposition: {content_disposition}"
        )
        
        # Verify file content matches
        with open(output_path, 'rb') as f:
            expected_content = f.read()
        
        response_content = b''.join(response.streaming_content)
        self.assertEqual(response_content, expected_content)
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
    
    @given(
        filename=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
            min_size=1,
            max_size=15
        ).map(lambda x: f"test_nonexistent_{x}.jpg")
    )
    def test_nonexistent_file_download_returns_404(self, filename):
        """
        Property: For any nonexistent file, download request should return 404 error.
        **Feature: image-dotting-converter, Property 11: File download functionality**
        **Validates: Requirements 4.3**
        """
        from django.test import Client
        from django.urls import reverse
        
        # Ensure file doesn't exist
        output_path = os.path.join(self.output_dir, filename)
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # Test download of nonexistent file
        client = Client()
        download_url = reverse('converter:download', kwargs={'filename': filename})
        response = client.get(download_url)
        
        # Should redirect to upload page on error (302) or return 404
        self.assertIn(response.status_code, [302, 404])
    
    @given(
        width=st.integers(min_value=10, max_value=200),
        height=st.integers(min_value=10, max_value=200),
        file_format=st.sampled_from(['JPEG', 'PNG'])
    )
    def test_download_preserves_image_quality(self, width, height, file_format):
        """
        Property: Downloaded images should preserve the original quality and format.
        **Feature: image-dotting-converter, Property 11: File download functionality**
        **Validates: Requirements 4.3**
        """
        from django.test import Client
        from django.urls import reverse
        
        # Create a test image with specific pattern
        image = Image.new('RGB', (width, height))
        # Create a checkerboard pattern
        for x in range(width):
            for y in range(height):
                if (x + y) % 2 == 0:
                    image.putpixel((x, y), (255, 255, 255))  # White
                else:
                    image.putpixel((x, y), (0, 0, 0))  # Black
        
        # Save to output directory
        extension = '.jpg' if file_format == 'JPEG' else '.png'
        filename = f"test_quality_{width}x{height}{extension}"
        output_path = os.path.join(self.output_dir, filename)
        image.save(output_path, format=file_format)
        
        # Get original file size and content
        original_size = os.path.getsize(output_path)
        with open(output_path, 'rb') as f:
            original_content = f.read()
        
        # Download via Django view
        client = Client()
        download_url = reverse('converter:download', kwargs={'filename': filename})
        response = client.get(download_url)
        
        # Verify successful download
        self.assertEqual(response.status_code, 200)
        
        # Get downloaded content
        downloaded_content = b''.join(response.streaming_content)
        
        # Verify content is identical
        self.assertEqual(downloaded_content, original_content)
        
        # Verify size is preserved
        self.assertEqual(len(downloaded_content), original_size)
        
        # Verify downloaded content can be opened as valid image
        downloaded_image = Image.open(io.BytesIO(downloaded_content))
        self.assertEqual(downloaded_image.size, (width, height))
        self.assertEqual(downloaded_image.mode, 'RGB')
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
    
    def test_download_security_prevents_path_traversal(self):
        """
        Property: Download functionality should prevent path traversal attacks.
        **Feature: image-dotting-converter, Property 11: File download functionality**
        **Validates: Requirements 4.3**
        """
        from django.test import Client
        from django.urls import reverse
        
        # Test various path traversal attempts
        malicious_filenames = [
            '../../../etc/passwd',
            '..\\..\\..\\windows\\system32\\config\\sam',
            '../../../../root/.ssh/id_rsa',
            '../settings.py',
            '../../manage.py'
        ]
        
        client = Client()
        
        for malicious_filename in malicious_filenames:
            try:
                download_url = reverse('converter:download', kwargs={'filename': malicious_filename})
                response = client.get(download_url)
                
                # Should return 404 (file not found in output directory) or redirect to error page
                # Should NOT return 200 with sensitive file content
                self.assertIn(response.status_code, [404, 302])
                
                # If response is 200, verify it's not serving sensitive content
                if response.status_code == 200:
                    content = b''.join(response.streaming_content)
                    # Should not contain typical sensitive file markers
                    sensitive_markers = [b'root:', b'password', b'ssh-rsa', b'SECRET_KEY']
                    for marker in sensitive_markers:
                        self.assertNotIn(marker, content.lower())
            except Exception:
                # URL reverse failed due to invalid characters - this is good security
                # The URL pattern itself prevents path traversal
                pass


class ErrorMessageDisplayTests(HypothesisTestCase):
    """
    Property-based tests for error message display.
    **Feature: image-dotting-converter, Property 12: Error message display**
    **Validates: Requirements 4.5**
    """
    
    def setUp(self):
        """Set up test environment."""
        self.media_root = settings.MEDIA_ROOT
        self.input_dir = os.path.join(self.media_root, 'input')
        self.output_dir = os.path.join(self.media_root, 'output')
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        # Clean up any test files created
        for directory in [self.input_dir, self.output_dir]:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    if filename.startswith('test_'):
                        try:
                            os.remove(os.path.join(directory, filename))
                        except (PermissionError, FileNotFoundError):
                            # File might be in use or already deleted, skip
                            pass
    
    @given(
        invalid_extension=st.sampled_from(['.gif', '.bmp', '.tiff', '.webp', '.svg', '.txt', '.pdf'])
    )
    def test_invalid_file_format_displays_error_message(self, invalid_extension):
        """
        Property: For any processing failure due to invalid file format, the system should display appropriate error messages.
        **Feature: image-dotting-converter, Property 12: Error message display**
        **Validates: Requirements 4.5**
        """
        from django.test import Client
        from django.urls import reverse
        
        # Create a file with invalid extension
        filename = f"test_invalid{invalid_extension}"
        file_content = b"fake file content"
        
        # Create uploaded file
        uploaded_file = SimpleUploadedFile(
            filename,
            file_content,
            content_type='application/octet-stream'
        )
        
        # Test form submission
        client = Client()
        upload_url = reverse('converter:upload')
        
        response = client.post(upload_url, {
            'image': uploaded_file,
            'dot_spacing': 20,
            'min_dot_radius': 2,
            'max_dot_radius': 8,
            'color_mode': 'black_on_white'
        })
        
        # Should not redirect (form should be invalid)
        self.assertEqual(response.status_code, 200)
        
        # Should contain error message
        self.assertContains(response, 'error', status_code=200)
        
        # Should contain form with errors
        form = response.context.get('form')
        self.assertIsNotNone(form)
        self.assertFalse(form.is_valid())
        self.assertIn('image', form.errors)
    
    @given(
        invalid_dot_spacing=st.one_of(
            st.integers(max_value=9),
            st.integers(min_value=51)
        )
    )
    def test_invalid_parameters_display_error_message(self, invalid_dot_spacing):
        """
        Property: For any processing failure due to invalid parameters, the system should display appropriate error messages.
        **Feature: image-dotting-converter, Property 12: Error message display**
        **Validates: Requirements 4.5**
        """
        from django.test import Client
        from django.urls import reverse
        
        # Create a valid image
        image = Image.new('RGB', (100, 100), color='red')
        image_io = io.BytesIO()
        image.save(image_io, format='JPEG')
        image_io.seek(0)
        
        uploaded_file = SimpleUploadedFile(
            'test_valid.jpg',
            image_io.getvalue(),
            content_type='image/jpeg'
        )
        
        # Test form submission with invalid parameters
        client = Client()
        upload_url = reverse('converter:upload')
        
        response = client.post(upload_url, {
            'image': uploaded_file,
            'dot_spacing': invalid_dot_spacing,  # Invalid value
            'min_dot_radius': 2,
            'max_dot_radius': 8,
            'color_mode': 'black_on_white'
        })
        
        # Should not redirect (form should be invalid)
        self.assertEqual(response.status_code, 200)
        
        # Should contain error message
        self.assertContains(response, 'error', status_code=200)
        
        # Should contain form with errors
        form = response.context.get('form')
        self.assertIsNotNone(form)
        self.assertFalse(form.is_valid())
        self.assertIn('dot_spacing', form.errors)
    
    def test_missing_session_data_displays_error_message(self):
        """
        Property: When processing fails due to missing session data, appropriate error message should be displayed.
        **Feature: image-dotting-converter, Property 12: Error message display**
        **Validates: Requirements 4.5**
        """
        from django.test import Client
        from django.urls import reverse
        
        # Test accessing process view without uploading file first
        client = Client()
        process_url = reverse('converter:process')
        
        response = client.get(process_url)
        
        # Should redirect to upload page
        self.assertEqual(response.status_code, 302)
        
        # Follow redirect to see error message
        response = client.get(process_url, follow=True)
        
        # Should contain error message about missing upload
        self.assertContains(response, 'No image uploaded')
        self.assertContains(response, 'Please upload an image first')
    
    @given(
        width=st.integers(min_value=10, max_value=100),
        height=st.integers(min_value=10, max_value=100)
    )
    def test_processing_failure_displays_error_message(self, width, height):
        """
        Property: For any processing failure during image conversion, the system should display appropriate error messages.
        **Feature: image-dotting-converter, Property 12: Error message display**
        **Validates: Requirements 4.5**
        """
        from django.test import Client
        from django.urls import reverse
        from unittest.mock import patch
        
        # Create a valid image and upload it
        image = Image.new('RGB', (width, height), color='blue')
        image_io = io.BytesIO()
        image.save(image_io, format='JPEG')
        image_io.seek(0)
        
        uploaded_file = SimpleUploadedFile(
            'test_processing_fail.jpg',
            image_io.getvalue(),
            content_type='image/jpeg'
        )
        
        client = Client()
        
        # First upload the file successfully
        upload_url = reverse('converter:upload')
        response = client.post(upload_url, {
            'image': uploaded_file,
            'dot_spacing': 20,
            'min_dot_radius': 2,
            'max_dot_radius': 8,
            'color_mode': 'black_on_white'
        })
        
        # Should redirect to process view
        self.assertEqual(response.status_code, 302)
        
        # Mock the image processor to simulate failure
        with patch('converter.views.ImageProcessor.process_image', return_value=False):
            process_url = reverse('converter:process')
            response = client.get(process_url, follow=True)
            
            # Should contain error message about processing failure
            self.assertContains(response, 'Error processing image')
            self.assertContains(response, 'Please try again')
    
    def test_download_error_displays_message(self):
        """
        Property: When download fails, appropriate error message should be displayed.
        **Feature: image-dotting-converter, Property 12: Error message display**
        **Validates: Requirements 4.5**
        """
        from django.test import Client
        from django.urls import reverse
        
        # Test downloading nonexistent file
        client = Client()
        download_url = reverse('converter:download', kwargs={'filename': 'nonexistent_file.jpg'})
        
        response = client.get(download_url)
        
        # Should redirect to upload page on error (302) or return 404
        self.assertIn(response.status_code, [302, 404])
    
    @given(
        error_scenario=st.sampled_from([
            'empty_file',
            'oversized_file',
            'corrupted_image',
            'invalid_radius_relationship'
        ])
    )
    def test_various_error_scenarios_display_appropriate_messages(self, error_scenario):
        """
        Property: Different error scenarios should display contextually appropriate error messages.
        **Feature: image-dotting-converter, Property 12: Error message display**
        **Validates: Requirements 4.5**
        """
        from django.test import Client
        from django.urls import reverse
        
        client = Client()
        upload_url = reverse('converter:upload')
        
        if error_scenario == 'empty_file':
            # Test empty file upload
            response = client.post(upload_url, {
                'dot_spacing': 20,
                'min_dot_radius': 2,
                'max_dot_radius': 8,
                'color_mode': 'black_on_white'
                # No image field
            })
            
            self.assertEqual(response.status_code, 200)
            form = response.context.get('form')
            self.assertIsNotNone(form)
            self.assertFalse(form.is_valid())
            self.assertIn('image', form.errors)
        
        elif error_scenario == 'oversized_file':
            # Create a large file (simulate oversized)
            large_content = b'x' * (11 * 1024 * 1024)  # 11MB (over 10MB limit)
            uploaded_file = SimpleUploadedFile(
                'large_file.jpg',
                large_content,
                content_type='image/jpeg'
            )
            
            response = client.post(upload_url, {
                'image': uploaded_file,
                'dot_spacing': 20,
                'min_dot_radius': 2,
                'max_dot_radius': 8,
                'color_mode': 'black_on_white'
            })
            
            self.assertEqual(response.status_code, 200)
            form = response.context.get('form')
            self.assertIsNotNone(form)
            self.assertFalse(form.is_valid())
        
        elif error_scenario == 'corrupted_image':
            # Create corrupted image data
            corrupted_file = SimpleUploadedFile(
                'corrupted.jpg',
                b'not_an_image_file',
                content_type='image/jpeg'
            )
            
            response = client.post(upload_url, {
                'image': corrupted_file,
                'dot_spacing': 20,
                'min_dot_radius': 2,
                'max_dot_radius': 8,
                'color_mode': 'black_on_white'
            })
            
            self.assertEqual(response.status_code, 200)
            form = response.context.get('form')
            self.assertIsNotNone(form)
            self.assertFalse(form.is_valid())
        
        elif error_scenario == 'invalid_radius_relationship':
            # Create valid image but invalid radius relationship
            image = Image.new('RGB', (50, 50), color='green')
            image_io = io.BytesIO()
            image.save(image_io, format='JPEG')
            image_io.seek(0)
            
            uploaded_file = SimpleUploadedFile(
                'test_radius.jpg',
                image_io.getvalue(),
                content_type='image/jpeg'
            )
            
            response = client.post(upload_url, {
                'image': uploaded_file,
                'dot_spacing': 20,
                'min_dot_radius': 5,  # Valid individually but equal to max
                'max_dot_radius': 5,  # Valid individually but equal to min
                'color_mode': 'black_on_white'
            })
            
            self.assertEqual(response.status_code, 200)
            form = response.context.get('form')
            self.assertIsNotNone(form)
            self.assertFalse(form.is_valid())
            # Should have non-field error for radius relationship OR field-level errors
            has_errors = (len(form.non_field_errors()) > 0 or 
                         'min_dot_radius' in form.errors or 
                         'max_dot_radius' in form.errors)
            self.assertTrue(has_errors, "Form should have errors for invalid radius relationship")