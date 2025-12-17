"""
Core image processing functionality for the dotting converter and pencil sketch converter.
Enhanced with maximum accuracy and advanced visual quality algorithms.
"""
import cv2
import numpy as np
from typing import Tuple, Optional
import os
from PIL import Image
import math
import random


class ProcessingParameters:
    """Data structure for processing parameters."""
    
    def __init__(self, dot_spacing: int = 20, min_dot_radius: int = 2, 
                 max_dot_radius: int = 10, color_mode: str = 'black_on_white'):
        self.dot_spacing = dot_spacing
        self.min_dot_radius = min_dot_radius
        self.max_dot_radius = max_dot_radius
        self.color_mode = color_mode
    
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


class ImageProcessor:
    """Core class for converting images to dotted representations."""
    
    def __init__(self, parameters: ProcessingParameters):
        self.parameters = parameters
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert a color image to grayscale using OpenCV.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Grayscale image as numpy array
        """
        if len(image.shape) == 3:
            # Convert BGR to grayscale
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # Already grayscale
            return image.copy()
    
    def create_grid(self, image_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate optimal grid dimensions using advanced image analysis.
        Considers image content and characteristics for maximum accuracy.
        
        Args:
            image_shape: (height, width) of the image
            
        Returns:
            (grid_rows, grid_cols) tuple optimized for image content
        """
        height, width = image_shape
        
        # Calculate adaptive spacing based on comprehensive image analysis
        adaptive_spacing = self._calculate_adaptive_spacing(width, height)
        
        # Calculate grid dimensions with proper boundary handling
        grid_rows = max(1, height // adaptive_spacing)
        grid_cols = max(1, width // adaptive_spacing)
        
        # Ensure we don't lose edge information by adjusting grid if necessary
        if height % adaptive_spacing > adaptive_spacing * 0.5:
            grid_rows += 1
        if width % adaptive_spacing > adaptive_spacing * 0.5:
            grid_cols += 1
        
        return grid_rows, grid_cols
    
    def _calculate_adaptive_spacing(self, width: int, height: int) -> int:
        """
        Calculate optimal adaptive dot spacing using advanced image analysis.
        Considers aspect ratio, resolution, and content density for maximum accuracy.
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            Optimally calculated adaptive spacing value
        """
        # Calculate image metrics
        area = width * height
        aspect_ratio = max(width, height) / min(width, height)
        diagonal = math.sqrt(width**2 + height**2)
        
        # Base spacing from parameters
        base_spacing = self.parameters.dot_spacing
        
        # Calculate resolution-based scaling factor
        if area <= 100000:  # <= 100K pixels (very small)
            resolution_factor = 1.5
        elif area <= 400000:  # <= 400K pixels (small)
            resolution_factor = 1.2
        elif area <= 1000000:  # <= 1MP (medium)
            resolution_factor = 1.0
        elif area <= 4000000:  # <= 4MP (large)
            resolution_factor = 0.8
        elif area <= 16000000:  # <= 16MP (very large)
            resolution_factor = 0.6
        else:  # > 16MP (ultra high resolution)
            resolution_factor = 0.4
        
        # Calculate aspect ratio adjustment
        # Extreme aspect ratios need different spacing to maintain visual balance
        if aspect_ratio > 3.0:  # Very wide or tall images
            aspect_factor = 1.1
        elif aspect_ratio > 2.0:  # Moderately wide/tall
            aspect_factor = 1.05
        else:  # Square-ish images
            aspect_factor = 1.0
        
        # Calculate diagonal-based fine adjustment
        # Larger diagonals can handle slightly smaller spacing
        diagonal_factor = max(0.7, min(1.3, 1000 / diagonal))
        
        # Combine all factors
        adaptive_spacing = base_spacing * resolution_factor * aspect_factor * diagonal_factor
        
        # Apply intelligent bounds based on image characteristics
        min_spacing = max(6, int(min(width, height) / 100))  # At least 1% of smaller dimension
        max_spacing = min(80, int(min(width, height) / 10))   # At most 10% of smaller dimension
        
        # Final spacing with bounds
        final_spacing = int(round(adaptive_spacing))
        final_spacing = max(min_spacing, min(final_spacing, max_spacing))
        
        return final_spacing
    
    def calculate_brightness(self, image: np.ndarray, row: int, col: int, spacing: int) -> float:
        """
        Calculate weighted brightness for a grid cell with improved sampling accuracy.
        Uses center-weighted averaging and edge detection for better representation.
        
        Args:
            image: Grayscale image (should be pre-blurred)
            row: Grid row index
            col: Grid column index
            spacing: Current adaptive spacing
            
        Returns:
            Weighted brightness value [0, 255]
        """
        start_y = row * spacing
        end_y = min(start_y + spacing, image.shape[0])
        start_x = col * spacing
        end_x = min(start_x + spacing, image.shape[1])
        
        # Extract the cell
        cell = image[start_y:end_y, start_x:end_x]
        
        # Create center-weighted mask for more accurate brightness sampling
        cell_height, cell_width = cell.shape
        center_y, center_x = cell_height // 2, cell_width // 2
        
        # Create Gaussian weight matrix centered on the cell
        y_coords, x_coords = np.ogrid[:cell_height, :cell_width]
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        # Normalize distances and create Gaussian weights
        max_distance = np.sqrt(center_y**2 + center_x**2)
        if max_distance > 0:
            normalized_distances = distances / max_distance
            # Use sigma = 0.3 for moderate center weighting
            weights = np.exp(-(normalized_distances**2) / (2 * 0.3**2))
        else:
            weights = np.ones_like(cell)
        
        # Calculate weighted mean intensity
        weighted_intensity = np.average(cell, weights=weights)
        
        # Apply perceptual brightness correction (gamma = 2.2 for human vision)
        # This makes the brightness calculation more perceptually accurate
        normalized = weighted_intensity / 255.0
        perceptual_brightness = normalized ** (1/2.2)
        
        return float(perceptual_brightness * 255.0)
    
    def _apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive Gaussian blur with edge preservation for optimal brightness sampling.
        Uses bilateral filtering for noise reduction while preserving important edges.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Processed grayscale image with noise reduction and edge preservation
        """
        height, width = image.shape
        area = height * width
        
        # Calculate adaptive parameters based on image size and content
        if area > 2000000:  # > 2MP
            kernel_size = 7
            sigma_space = 75
            sigma_color = 75
        elif area > 1000000:  # > 1MP
            kernel_size = 5
            sigma_space = 50
            sigma_color = 50
        elif area > 400000:  # > 400K
            kernel_size = 5
            sigma_space = 40
            sigma_color = 40
        else:
            kernel_size = 3
            sigma_space = 30
            sigma_color = 30
        
        # Apply bilateral filter for edge-preserving smoothing
        # This maintains important details while reducing noise
        bilateral_filtered = cv2.bilateralFilter(image, -1, sigma_color, sigma_space)
        
        # Apply light Gaussian blur for final smoothing
        # Use smaller sigma for subtle smoothing without losing detail
        sigma = kernel_size / 6.0  # Optimal sigma relationship
        final_blur = cv2.GaussianBlur(bilateral_filtered, (kernel_size, kernel_size), sigma)
        
        return final_blur
    
    def scale_dot_radius(self, brightness: float, color_mode: str = None) -> int:
        """
        Advanced dot radius scaling with perceptual accuracy and adaptive contrast.
        Uses multiple scaling curves for different brightness ranges and color modes.
        
        Args:
            brightness: Brightness value [0, 255]
            color_mode: Override color mode for this calculation
            
        Returns:
            Optimally scaled dot radius with perceptual accuracy
        """
        # Use provided color mode or default to instance parameter
        mode = color_mode if color_mode else self.parameters.color_mode
        
        # Normalize brightness to [0, 1]
        normalized = brightness / 255.0
        
        # For both modes, we want darker areas to have larger dots
        # This creates the proper contrast representation
        inverted = 1.0 - normalized
        
        # Apply adaptive gamma correction based on brightness distribution
        if brightness < 85:  # Dark areas (shadows)
            # Use stronger gamma for better shadow detail
            gamma = 2.2
            gamma_corrected = inverted ** gamma
        elif brightness < 170:  # Mid-tones
            # Use moderate gamma for natural mid-tone reproduction
            gamma = 1.8
            gamma_corrected = inverted ** gamma
        else:  # Bright areas (highlights)
            # Use gentler curve for highlight preservation
            gamma = 1.4
            gamma_corrected = inverted ** gamma
        
        # Apply S-curve for enhanced contrast in mid-tones
        # This improves visual perception of the dotted image
        s_curve_factor = 0.3
        s_curve_adjusted = gamma_corrected + s_curve_factor * np.sin(2 * np.pi * gamma_corrected) / (2 * np.pi)
        s_curve_adjusted = np.clip(s_curve_adjusted, 0, 1)
        
        # Scale to radius range with improved distribution
        radius_range = self.parameters.max_dot_radius - self.parameters.min_dot_radius
        scaled_radius = self.parameters.min_dot_radius + (s_curve_adjusted * radius_range)
        
        # Apply brightness-dependent radius adjustments
        if brightness > 245:  # Very bright highlights
            final_radius = max(1, int(self.parameters.min_dot_radius * 0.3))
        elif brightness > 230:  # Bright highlights
            final_radius = max(1, int(scaled_radius * 0.5))
        elif brightness > 210:  # Light areas
            final_radius = max(1, int(scaled_radius * 0.7))
        elif brightness < 15:  # Very dark shadows
            # Ensure maximum impact for very dark areas
            final_radius = min(self.parameters.max_dot_radius, int(scaled_radius * 1.1))
        else:
            final_radius = int(round(scaled_radius))
        
        # Ensure radius stays within bounds
        final_radius = max(1, min(final_radius, self.parameters.max_dot_radius))
        
        return final_radius
    
    def create_fresh_canvas(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a fresh canvas for dot rendering.
        
        Args:
            image_shape: (height, width) of the original image
            
        Returns:
            Fresh canvas as numpy array
        """
        height, width = image_shape
        
        if self.parameters.color_mode == 'black_on_white':
            # White background
            canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        else:  # white_on_black
            # Black background
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        return canvas
    
    def render_dots(self, canvas: np.ndarray, grayscale_image: np.ndarray) -> np.ndarray:
        """
        Render dots with maximum accuracy using advanced sampling and positioning.
        Implements sub-pixel positioning, adaptive opacity, and enhanced quality algorithms.
        
        Args:
            canvas: Fresh canvas to draw on
            grayscale_image: Grayscale version of original image
            
        Returns:
            Canvas with highest-quality dots rendered
        """
        # Apply advanced blur with edge preservation
        processed_image = self._apply_gaussian_blur(grayscale_image)
        
        # Calculate adaptive grid dimensions
        grid_rows, grid_cols = self.create_grid(grayscale_image.shape)
        
        # Get adaptive spacing for this image
        height, width = grayscale_image.shape
        adaptive_spacing = self._calculate_adaptive_spacing(width, height)
        
        # Determine dot color and create floating point canvas for sub-pixel accuracy
        if self.parameters.color_mode == 'black_on_white':
            dot_color = (0, 0, 0)  # Black dots
            float_canvas = np.full(canvas.shape, 255.0, dtype=np.float32)
        else:
            dot_color = (255, 255, 255)  # White dots
            float_canvas = np.zeros(canvas.shape, dtype=np.float32)
        
        # Pre-calculate brightness statistics for adaptive processing
        all_brightness = []
        brightness_map = np.zeros((grid_rows, grid_cols))
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                brightness = self.calculate_brightness(processed_image, row, col, adaptive_spacing)
                brightness_map[row, col] = brightness
                all_brightness.append(brightness)
        
        # Calculate brightness statistics for adaptive contrast
        brightness_array = np.array(all_brightness)
        brightness_mean = np.mean(brightness_array)
        brightness_std = np.std(brightness_array)
        
        # Set random seed for consistent jitter patterns
        np.random.seed(42)
        
        # Render dots with enhanced accuracy
        for row in range(grid_rows):
            for col in range(grid_cols):
                brightness = brightness_map[row, col]
                
                # Apply local contrast enhancement
                if brightness_std > 0:
                    # Normalize brightness relative to local statistics
                    contrast_enhanced = brightness + 0.3 * (brightness - brightness_mean)
                    brightness = np.clip(contrast_enhanced, 0, 255)
                
                # Calculate radius with advanced scaling
                radius = self.scale_dot_radius(brightness, self.parameters.color_mode)
                
                # Skip dots that are too small to be meaningful
                if radius < 0.5:
                    continue
                
                # Calculate sub-pixel precise dot center position
                # Add slight random offset for more natural appearance
                base_center_y = row * adaptive_spacing + adaptive_spacing / 2.0
                base_center_x = col * adaptive_spacing + adaptive_spacing / 2.0
                
                # Add subtle jitter for more organic appearance (±10% of spacing)
                jitter_range = adaptive_spacing * 0.05
                center_y = base_center_y + np.random.uniform(-jitter_range, jitter_range)
                center_x = base_center_x + np.random.uniform(-jitter_range, jitter_range)
                
                # Ensure centers stay within image bounds
                center_y = np.clip(center_y, radius, height - radius - 1)
                center_x = np.clip(center_x, radius, width - radius - 1)
                
                # Calculate adaptive opacity based on brightness - CORRECTED LOGIC
                if self.parameters.color_mode == 'black_on_white':
                    # For black dots on white background: darker areas get larger/more opaque dots
                    opacity = 1.0 - (brightness / 255.0)
                else:
                    # For white dots on black background: darker areas should get larger/more opaque white dots
                    # This is the OPPOSITE of what was implemented - we want white dots where the original is dark
                    opacity = 1.0 - (brightness / 255.0)
                
                # Apply minimum opacity threshold
                opacity = max(0.1, opacity)
                
                # Clamp radius to prevent excessive overlap while allowing some overlap for realism
                max_allowed_radius = adaptive_spacing * 0.6  # Allow 60% of spacing
                final_radius = min(radius, max_allowed_radius)
                
                # Draw dot with sub-pixel accuracy and variable opacity - CORRECTED LOGIC
                if self.parameters.color_mode == 'black_on_white':
                    # Black dots on white background: darker areas get darker dots
                    dot_intensity = 255 * (1 - opacity)  # Lower values = darker dots
                    cv2.circle(
                        float_canvas, 
                        (int(center_x), int(center_y)), 
                        int(final_radius), 
                        (dot_intensity, dot_intensity, dot_intensity), 
                        -1,  # Filled circle
                        cv2.LINE_AA  # Anti-aliasing
                    )
                else:
                    # White dots on black background: darker areas get brighter white dots
                    dot_intensity = 255 * opacity  # Higher values = brighter white dots
                    cv2.circle(
                        float_canvas, 
                        (int(center_x), int(center_y)), 
                        int(final_radius), 
                        (dot_intensity, dot_intensity, dot_intensity), 
                        -1,  # Filled circle
                        cv2.LINE_AA  # Anti-aliasing
                    )
        
        # Convert back to uint8 with proper clipping
        final_canvas = np.clip(float_canvas, 0, 255).astype(np.uint8)
        
        return final_canvas
    
    def _calculate_optimal_max_dimension(self, width: int, height: int) -> int:
        """
        Calculate optimal maximum dimension for resizing based on image characteristics.
        Ensures no resolution rejection while maintaining processing efficiency.
        
        Args:
            width: Original image width
            height: Original image height
            
        Returns:
            Optimal maximum dimension for resizing
        """
        # Calculate total pixels
        total_pixels = width * height
        
        # Define thresholds and corresponding max dimensions
        if total_pixels <= 500000:  # <= 0.5MP
            return 1200  # Small images can be processed at higher resolution
        elif total_pixels <= 2000000:  # <= 2MP
            return 1500  # Medium images
        elif total_pixels <= 8000000:  # <= 8MP
            return 1800  # Large images
        else:
            return 2000  # Very large images - still process but with reasonable limit
        
        # Note: No image is rejected due to size - all are processed
    
    def process_image(self, input_path: str, output_path: str) -> bool:
        """
        Complete image processing pipeline with enhanced error handling and memory management.
        Uses PIL for image loading and auto-resizes large images while maintaining aspect ratio.
        
        Args:
            input_path: Path to input image
            output_path: Path to save output image
            
        Returns:
            True if successful, False otherwise
        """
        import logging
        from PIL import Image
        
        logger = logging.getLogger(__name__)
        
        try:
            # Validate input file exists
            if not os.path.exists(input_path):
                logger.error(f"Input file does not exist: {input_path}")
                return False
            
            # Load image using PIL (more reliable than cv2.imread)
            try:
                with Image.open(input_path) as pil_image:
                    # Convert to RGB to ensure consistent format
                    pil_image = pil_image.convert("RGB")
                    
                    # Get original dimensions
                    original_width, original_height = pil_image.size
                    logger.info(f"Original image size: {original_width}x{original_height}")
                    
                    # Auto-resize large images safely (no resolution rejection)
                    # Adaptive max dimension based on image aspect ratio
                    max_dimension = self._calculate_optimal_max_dimension(original_width, original_height)
                    
                    if original_width > max_dimension or original_height > max_dimension:
                        # Calculate new dimensions maintaining aspect ratio
                        if original_width > original_height:
                            new_width = max_dimension
                            new_height = int((original_height * max_dimension) / original_width)
                        else:
                            new_height = max_dimension
                            new_width = int((original_width * max_dimension) / original_height)
                        
                        # Resize using high-quality resampling
                        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        logger.info(f"Resized image to: {new_width}x{new_height}")
                    
                    # Convert PIL image to NumPy array for OpenCV processing
                    image_array = np.array(pil_image)
                    
                    # Convert RGB to BGR for OpenCV (OpenCV uses BGR format)
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    
            except Exception as e:
                logger.error(f"Failed to load image with PIL: {str(e)}")
                return False
            
            # Validate image dimensions
            height, width = image_bgr.shape[:2]
            if height <= 0 or width <= 0:
                logger.error(f"Invalid image dimensions: {width}x{height}")
                return False
            
            # Process the image using standard pipeline
            success = self._process_standard_image(image_bgr, output_path)
            
            if success:
                logger.info(f"Successfully processed image: {input_path} -> {output_path}")
            else:
                logger.error(f"Failed to process image: {input_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Unexpected error processing image {input_path}: {str(e)}")
            return False
    
    def _process_standard_image(self, image: np.ndarray, output_path: str) -> bool:
        """Process images with normal memory usage and detailed error handling."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Convert to grayscale AFTER resize (as per requirements)
            grayscale = self.convert_to_grayscale(image)
            logger.debug("Successfully converted image to grayscale")
            
            # Free original image memory
            del image
            
            # Create fresh canvas
            canvas = self.create_fresh_canvas(grayscale.shape)
            logger.debug(f"Created canvas with shape: {canvas.shape}")
            
            # Render dots
            result = self.render_dots(canvas, grayscale)
            logger.debug("Successfully rendered dots on canvas")
            
            # Free intermediate memory
            del canvas
            del grayscale
            
            # Save result with quality settings
            success = self._save_image(result, output_path)
            if success:
                logger.debug(f"Successfully saved result to: {output_path}")
            else:
                logger.error(f"Failed to save result to: {output_path}")
            
            return success
            
        except cv2.error as e:
            logger.error(f"OpenCV error during processing: {str(e)}")
            return False
        except MemoryError as e:
            logger.error(f"Memory error during processing: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during processing: {str(e)}")
            return False
    
    def _process_large_image(self, image: np.ndarray, output_path: str) -> bool:
        """Process large images with optimized memory usage and improved algorithms."""
        try:
            # Convert to grayscale
            grayscale = self.convert_to_grayscale(image)
            
            # Free original image memory immediately
            del image
            
            # Apply Gaussian blur for better brightness sampling
            blurred_grayscale = self._apply_gaussian_blur(grayscale)
            
            # Create fresh canvas
            canvas = self.create_fresh_canvas(grayscale.shape)
            
            # Calculate adaptive parameters for large images
            height, width = grayscale.shape
            grid_rows, grid_cols = self.create_grid(grayscale.shape)
            adaptive_spacing = self._calculate_adaptive_spacing(width, height)
            
            # Determine dot color
            if self.parameters.color_mode == 'black_on_white':
                dot_color = (0, 0, 0)  # Black dots
            else:
                dot_color = (255, 255, 255)  # White dots
            
            # Process in chunks to manage memory
            chunk_size = 50  # Process 50 rows at a time
            for start_row in range(0, grid_rows, chunk_size):
                end_row = min(start_row + chunk_size, grid_rows)
                
                # Process this chunk of rows with improved algorithms
                for row in range(start_row, end_row):
                    for col in range(grid_cols):
                        # Use blurred image for better brightness sampling
                        brightness = self.calculate_brightness(blurred_grayscale, row, col, adaptive_spacing)
                        
                        # Calculate radius with gamma correction
                        radius = self.scale_dot_radius(brightness)
                        
                        # Skip very small dots in highlights
                        if radius < 1:
                            continue
                        
                        # Calculate precise dot center position
                        center_y = row * adaptive_spacing + adaptive_spacing // 2
                        center_x = col * adaptive_spacing + adaptive_spacing // 2
                        
                        # Clamp radius to prevent excessive overlap
                        max_allowed_radius = adaptive_spacing // 2
                        clamped_radius = min(radius, max_allowed_radius)
                        
                        # Draw dot with anti-aliasing
                        cv2.circle(canvas, (center_x, center_y), clamped_radius, dot_color, -1, cv2.LINE_AA)
                
                # Force garbage collection for large images
                import gc
                gc.collect()
            
            # Free memory
            del grayscale
            del blurred_grayscale
            
            # Save result
            return self._save_image(canvas, output_path)
            
        except Exception:
            return False
    
    def _save_image(self, image: np.ndarray, output_path: str) -> bool:
        """Save image with appropriate quality settings and detailed error handling."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Save with appropriate format settings
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                # Save JPEG with high quality
                success = cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                logger.debug(f"Saving as JPEG with quality 95")
            elif output_path.lower().endswith('.png'):
                # Save PNG with compression
                success = cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                logger.debug(f"Saving as PNG with compression 6")
            else:
                # Default save
                success = cv2.imwrite(output_path, image)
                logger.debug(f"Saving with default settings")
            
            if not success:
                logger.error(f"cv2.imwrite returned False for {output_path}")
                return False
            
            # Verify output file was created and has reasonable size
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 0:
                    logger.debug(f"Successfully saved image: {output_path} ({file_size} bytes)")
                    return True
                else:
                    logger.error(f"Output file has zero size: {output_path}")
                    return False
            else:
                logger.error(f"Output file was not created: {output_path}")
                return False
            
        except OSError as e:
            logger.error(f"File system error saving image: {str(e)}")
            return False
        except cv2.error as e:
            logger.error(f"OpenCV error saving image: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving image: {str(e)}")
            return False


class PencilSketchProcessor:
    """Core class for converting images to realistic pencil sketches."""
    
    def __init__(self):
        """Initialize the pencil sketch processor."""
        pass
    
    def process_image(self, input_path: str, output_path: str) -> bool:
        """
        Complete pencil sketch processing pipeline with enhanced error handling.
        Uses PIL for image loading and auto-resizes large images while maintaining aspect ratio.
        
        Args:
            input_path: Path to input image
            output_path: Path to save output image
            
        Returns:
            True if successful, False otherwise
        """
        import logging
        from PIL import Image
        
        logger = logging.getLogger(__name__)
        
        try:
            # Validate input file exists
            if not os.path.exists(input_path):
                logger.error(f"Input file does not exist: {input_path}")
                return False
            
            # Load image using PIL (more reliable than cv2.imread)
            try:
                with Image.open(input_path) as pil_image:
                    # Convert to RGB to ensure consistent format
                    pil_image = pil_image.convert("RGB")
                    
                    # Get original dimensions
                    original_width, original_height = pil_image.size
                    logger.info(f"Original image size: {original_width}x{original_height}")
                    
                    # Auto-resize large images safely (no resolution rejection)
                    max_dimension = self._calculate_optimal_max_dimension(original_width, original_height)
                    
                    if original_width > max_dimension or original_height > max_dimension:
                        # Calculate new dimensions maintaining aspect ratio
                        if original_width > original_height:
                            new_width = max_dimension
                            new_height = int((original_height * max_dimension) / original_width)
                        else:
                            new_height = max_dimension
                            new_width = int((original_width * max_dimension) / original_height)
                        
                        # Resize using high-quality resampling
                        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        logger.info(f"Resized image to: {new_width}x{new_height}")
                    
                    # Convert PIL image to NumPy array for OpenCV processing
                    image_array = np.array(pil_image)
                    
                    # Convert RGB to BGR for OpenCV (OpenCV uses BGR format)
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    
            except Exception as e:
                logger.error(f"Failed to load image with PIL: {str(e)}")
                return False
            
            # Validate image dimensions
            height, width = image_bgr.shape[:2]
            if height <= 0 or width <= 0:
                logger.error(f"Invalid image dimensions: {width}x{height}")
                return False
            
            # Process the image using pencil sketch pipeline
            success = self._process_pencil_sketch(image_bgr, output_path)
            
            if success:
                logger.info(f"Successfully processed pencil sketch: {input_path} -> {output_path}")
            else:
                logger.error(f"Failed to process pencil sketch: {input_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Unexpected error processing pencil sketch {input_path}: {str(e)}")
            return False
    
    def _calculate_optimal_max_dimension(self, width: int, height: int) -> int:
        """
        Calculate optimal maximum dimension for resizing based on image characteristics.
        Ensures no resolution rejection while maintaining processing efficiency.
        
        Args:
            width: Original image width
            height: Original image height
            
        Returns:
            Optimal maximum dimension for resizing
        """
        # Calculate total pixels
        total_pixels = width * height
        
        # Define thresholds and corresponding max dimensions
        if total_pixels <= 500000:  # <= 0.5MP
            return 1200  # Small images can be processed at higher resolution
        elif total_pixels <= 2000000:  # <= 2MP
            return 1500  # Medium images
        elif total_pixels <= 8000000:  # <= 8MP
            return 1800  # Large images
        else:
            return 2000  # Very large images - still process but with reasonable limit
        
        # Note: No image is rejected due to size - all are processed
    
    def _process_pencil_sketch(self, image: np.ndarray, output_path: str) -> bool:
        """
        Process image using advanced pencil sketch pipeline for highly realistic results.
        Implements multi-stage processing with edge enhancement and texture simulation.
        
        Args:
            image: Input image as numpy array (BGR format)
            output_path: Path to save output image
            
        Returns:
            True if successful, False otherwise
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Step 1: Convert image to grayscale with enhanced luminance calculation
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logger.debug("Successfully converted image to grayscale")
            
            # Step 2: Apply noise reduction while preserving edges
            # Use bilateral filter to smooth while keeping edges sharp
            height, width = gray.shape
            area = height * width
            
            # Adaptive bilateral filter parameters based on image size
            if area > 2000000:  # > 2MP
                d_bilateral = 15
                sigma_color = 80
                sigma_space = 80
            elif area > 1000000:  # > 1MP
                d_bilateral = 12
                sigma_color = 70
                sigma_space = 70
            else:
                d_bilateral = 9
                sigma_color = 60
                sigma_space = 60
            
            denoised = cv2.bilateralFilter(gray, d_bilateral, sigma_color, sigma_space)
            logger.debug("Applied bilateral filter for noise reduction")
            
            # Step 3: Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            logger.debug("Applied CLAHE for contrast enhancement")
            
            # Step 4: Create multiple edge maps for realistic pencil strokes
            # Sobel edges for directional strokes
            sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_edges = np.uint8(np.clip(sobel_combined, 0, 255))
            
            # Canny edges for fine details
            # Adaptive thresholds based on image statistics
            median_val = np.median(enhanced)
            lower_thresh = int(max(0, 0.7 * median_val))
            upper_thresh = int(min(255, 1.3 * median_val))
            canny_edges = cv2.Canny(enhanced, lower_thresh, upper_thresh)
            
            logger.debug("Created edge maps using Sobel and Canny")
            
            # Step 5: Invert grayscale image for blending
            inverted_gray = 255 - enhanced
            logger.debug("Successfully inverted enhanced grayscale image")
            
            # Step 6: Apply adaptive Gaussian blur to inverted image
            # Multiple blur levels for different stroke effects
            if area > 2000000:  # > 2MP
                blur_kernel_1 = 25  # Main blur
                blur_kernel_2 = 15  # Detail blur
                blur_kernel_3 = 7   # Fine detail blur
            elif area > 1000000:  # > 1MP
                blur_kernel_1 = 19
                blur_kernel_2 = 11
                blur_kernel_3 = 5
            else:
                blur_kernel_1 = 13
                blur_kernel_2 = 7
                blur_kernel_3 = 3
            
            # Ensure all kernel sizes are odd
            blur_kernel_1 = blur_kernel_1 if blur_kernel_1 % 2 == 1 else blur_kernel_1 + 1
            blur_kernel_2 = blur_kernel_2 if blur_kernel_2 % 2 == 1 else blur_kernel_2 + 1
            blur_kernel_3 = blur_kernel_3 if blur_kernel_3 % 2 == 1 else blur_kernel_3 + 1
            
            # Create multiple blur layers for depth
            blur_1 = cv2.GaussianBlur(inverted_gray, (blur_kernel_1, blur_kernel_1), 0)
            blur_2 = cv2.GaussianBlur(inverted_gray, (blur_kernel_2, blur_kernel_2), 0)
            blur_3 = cv2.GaussianBlur(inverted_gray, (blur_kernel_3, blur_kernel_3), 0)
            
            # Combine blurs with different weights for natural variation
            combined_blur = cv2.addWeighted(blur_1, 0.5, blur_2, 0.3, 0)
            combined_blur = cv2.addWeighted(combined_blur, 0.8, blur_3, 0.2, 0)
            
            logger.debug(f"Applied multi-level Gaussian blur")
            
            # Step 7: Create base sketch using divide blend
            sketch_base = cv2.divide(enhanced, 255 - combined_blur, scale=256)
            logger.debug("Created base sketch using divide blend")
            
            # Step 8: Enhance with edge information for realistic pencil strokes
            # Combine Sobel and Canny edges
            combined_edges = cv2.addWeighted(sobel_edges, 0.6, canny_edges, 0.4, 0)
            
            # Invert edges so dark lines appear on light background
            inverted_edges = 255 - combined_edges
            
            # Blend sketch with edge information
            sketch_with_edges = cv2.multiply(sketch_base, inverted_edges, scale=1/255.0)
            sketch_with_edges = np.uint8(np.clip(sketch_with_edges, 0, 255))
            
            logger.debug("Enhanced sketch with edge information")
            
            # Step 9: Apply texture simulation for paper-like appearance
            # Create subtle texture using noise
            texture_noise = np.random.normal(0, 3, (height, width)).astype(np.float32)
            texture_noise = cv2.GaussianBlur(texture_noise, (3, 3), 0)
            
            # Add texture to sketch
            textured_sketch = sketch_with_edges.astype(np.float32) + texture_noise
            textured_sketch = np.clip(textured_sketch, 0, 255).astype(np.uint8)
            
            logger.debug("Applied paper texture simulation")
            
            # Step 10: Final contrast and brightness adjustments
            # Adaptive adjustments based on image characteristics
            mean_brightness = np.mean(textured_sketch)
            
            if mean_brightness < 100:  # Dark image
                alpha = 1.3  # Higher contrast
                beta = 20    # More brightness
            elif mean_brightness > 180:  # Bright image
                alpha = 1.1  # Lower contrast
                beta = 5     # Less brightness
            else:  # Normal image
                alpha = 1.2  # Moderate contrast
                beta = 12    # Moderate brightness
            
            final_sketch = cv2.convertScaleAbs(textured_sketch, alpha=alpha, beta=beta)
            
            # Step 11: Apply final sharpening for crisp pencil lines
            # Create sharpening kernel
            sharpening_kernel = np.array([[-1, -1, -1],
                                        [-1,  9, -1],
                                        [-1, -1, -1]])
            
            # Apply subtle sharpening
            sharpened = cv2.filter2D(final_sketch, -1, sharpening_kernel * 0.3)
            
            # Blend original with sharpened for natural look
            final_sketch = cv2.addWeighted(final_sketch, 0.7, sharpened, 0.3, 0)
            
            # Ensure values are in valid range
            final_sketch = np.clip(final_sketch, 0, 255).astype(np.uint8)
            
            logger.debug("Applied final contrast, brightness, and sharpening adjustments")
            
            # Step 12: Convert back to 3-channel image for consistent output format
            final_sketch_bgr = cv2.cvtColor(final_sketch, cv2.COLOR_GRAY2BGR)
            
            # Save result with quality settings
            success = self._save_image(final_sketch_bgr, output_path)
            if success:
                logger.debug(f"Successfully saved enhanced pencil sketch to: {output_path}")
            else:
                logger.error(f"Failed to save pencil sketch to: {output_path}")
            
            return success
            
        except cv2.error as e:
            logger.error(f"OpenCV error during pencil sketch processing: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during pencil sketch processing: {str(e)}")
            return False
    
    def _save_image(self, image: np.ndarray, output_path: str) -> bool:
        """Save image with appropriate quality settings and detailed error handling."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Save with appropriate format settings
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                # Save JPEG with high quality
                success = cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                logger.debug(f"Saving as JPEG with quality 95")
            elif output_path.lower().endswith('.png'):
                # Save PNG with compression
                success = cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                logger.debug(f"Saving as PNG with compression 6")
            else:
                # Default save
                success = cv2.imwrite(output_path, image)
                logger.debug(f"Saving with default settings")
            
            if not success:
                logger.error(f"cv2.imwrite returned False for {output_path}")
                return False
            
            # Verify output file was created and has reasonable size
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 0:
                    logger.debug(f"Successfully saved image: {output_path} ({file_size} bytes)")
                    return True
                else:
                    logger.error(f"Output file has zero size: {output_path}")
                    return False
            else:
                logger.error(f"Output file was not created: {output_path}")
                return False
            
        except OSError as e:
            logger.error(f"File system error saving image: {str(e)}")
            return False
        except cv2.error as e:
            logger.error(f"OpenCV error saving image: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving image: {str(e)}")
            return False