"""
Core image processing functionality for the dotting converter.
Enhanced with improved accuracy and visual quality algorithms.
"""
import cv2
import numpy as np
from typing import Tuple, Optional
import os
from PIL import Image
import math


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
        Calculate grid dimensions with adaptive dot spacing based on image size.
        Implements adaptive dot density for better visual quality.
        
        Args:
            image_shape: (height, width) of the image
            
        Returns:
            (grid_rows, grid_cols) tuple
        """
        height, width = image_shape
        
        # Calculate adaptive spacing based on image size
        adaptive_spacing = self._calculate_adaptive_spacing(width, height)
        
        grid_rows = height // adaptive_spacing
        grid_cols = width // adaptive_spacing
        return grid_rows, grid_cols
    
    def _calculate_adaptive_spacing(self, width: int, height: int) -> int:
        """
        Calculate adaptive dot spacing based on image dimensions.
        Smaller spacing for large images, larger spacing for small images.
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            Adaptive spacing value
        """
        # Calculate image area
        area = width * height
        
        # Base spacing from parameters
        base_spacing = self.parameters.dot_spacing
        
        # Define size thresholds and scaling factors
        small_threshold = 200 * 200      # 40K pixels
        medium_threshold = 800 * 800     # 640K pixels
        large_threshold = 1200 * 1200    # 1.44M pixels
        
        if area <= small_threshold:
            # Small images: increase spacing to avoid overcrowding
            adaptive_spacing = int(base_spacing * 1.3)
        elif area <= medium_threshold:
            # Medium images: use base spacing
            adaptive_spacing = base_spacing
        elif area <= large_threshold:
            # Large images: reduce spacing for more detail
            adaptive_spacing = int(base_spacing * 0.8)
        else:
            # Very large images: significantly reduce spacing
            adaptive_spacing = int(base_spacing * 0.6)
        
        # Ensure spacing stays within reasonable bounds
        adaptive_spacing = max(8, min(adaptive_spacing, 60))
        
        return adaptive_spacing
    
    def calculate_brightness(self, image: np.ndarray, row: int, col: int, spacing: int) -> float:
        """
        Calculate average brightness for a grid cell with Gaussian blur preprocessing.
        Implements better brightness sampling for improved accuracy.
        
        Args:
            image: Grayscale image (should be pre-blurred)
            row: Grid row index
            col: Grid column index
            spacing: Current adaptive spacing
            
        Returns:
            Average brightness value [0, 255]
        """
        start_y = row * spacing
        end_y = min(start_y + spacing, image.shape[0])
        start_x = col * spacing
        end_x = min(start_x + spacing, image.shape[1])
        
        # Extract the cell
        cell = image[start_y:end_y, start_x:end_x]
        
        # Calculate mean intensity of the entire cell (not just a single pixel)
        # This provides better representation of the area's brightness
        mean_intensity = float(np.mean(cell))
        
        return mean_intensity
    
    def _apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to the image before brightness sampling.
        This reduces noise and provides smoother brightness transitions.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Blurred grayscale image
        """
        # Calculate kernel size based on image dimensions
        # Larger images get slightly more blur for better sampling
        height, width = image.shape
        area = height * width
        
        if area > 1000000:  # > 1MP
            kernel_size = 5
        elif area > 400000:  # > 400K
            kernel_size = 3
        else:
            kernel_size = 3
        
        # Apply Gaussian blur with calculated kernel size
        # Sigma is automatically calculated by OpenCV when set to 0
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return blurred
    
    def scale_dot_radius(self, brightness: float) -> int:
        """
        Scale dot radius using non-linear gamma correction for better contrast.
        Implements improved radius calculation with gamma scaling.
        
        Args:
            brightness: Brightness value [0, 255]
            
        Returns:
            Dot radius scaled between min and max radius with gamma correction
        """
        # Normalize brightness to [0, 1]
        normalized = brightness / 255.0
        
        # Invert for darker areas = larger dots
        inverted = 1.0 - normalized
        
        # Apply gamma correction for better contrast
        # Use gamma = 1.8 for optimal visual balance
        gamma = 1.8
        gamma_corrected = inverted ** gamma
        
        # Scale to radius range
        radius_range = self.parameters.max_dot_radius - self.parameters.min_dot_radius
        scaled_radius = self.parameters.min_dot_radius + (gamma_corrected * radius_range)
        
        # Round to integer and ensure minimum radius for very bright areas
        final_radius = int(round(scaled_radius))
        
        # For very bright areas (highlights), use very small dots or no dots
        if brightness > 240:  # Very bright highlights
            final_radius = max(1, self.parameters.min_dot_radius // 2)
        elif brightness > 220:  # Bright areas
            final_radius = max(1, int(final_radius * 0.7))
        
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
        Render dots with improved accuracy, anti-aliasing, and precise placement.
        Implements all visual quality improvements.
        
        Args:
            canvas: Fresh canvas to draw on
            grayscale_image: Grayscale version of original image
            
        Returns:
            Canvas with high-quality dots rendered
        """
        # Apply Gaussian blur for better brightness sampling
        blurred_image = self._apply_gaussian_blur(grayscale_image)
        
        # Calculate adaptive grid dimensions
        grid_rows, grid_cols = self.create_grid(grayscale_image.shape)
        
        # Get adaptive spacing for this image
        height, width = grayscale_image.shape
        adaptive_spacing = self._calculate_adaptive_spacing(width, height)
        
        # Determine dot color
        if self.parameters.color_mode == 'black_on_white':
            dot_color = (0, 0, 0)  # Black dots
        else:
            dot_color = (255, 255, 255)  # White dots
        
        # Render dots with improved algorithms
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Calculate brightness using blurred image for better sampling
                brightness = self.calculate_brightness(blurred_image, row, col, adaptive_spacing)
                
                # Calculate radius with gamma correction
                radius = self.scale_dot_radius(brightness)
                
                # Skip very small dots in highlights to preserve bright areas
                if radius < 1:
                    continue
                
                # Calculate precise dot center position (true cell center)
                center_y = row * adaptive_spacing + adaptive_spacing // 2
                center_x = col * adaptive_spacing + adaptive_spacing // 2
                
                # Clamp radius to prevent excessive overlap
                max_allowed_radius = adaptive_spacing // 2
                clamped_radius = min(radius, max_allowed_radius)
                
                # Draw dot with anti-aliasing for smooth edges
                cv2.circle(
                    canvas, 
                    (center_x, center_y), 
                    clamped_radius, 
                    dot_color, 
                    -1,  # Filled circle
                    cv2.LINE_AA  # Anti-aliasing for smooth edges
                )
        
        return canvas
    
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