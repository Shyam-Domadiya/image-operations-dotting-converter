# Image Operations - Dotting Converter

A Django web application that converts regular images into artistic dotted representations using advanced image processing algorithms.

## Features

- **Adaptive Dot Spacing**: Automatically adjusts dot density based on image size
- **Gamma Correction**: Enhanced contrast using non-linear gamma scaling
- **Anti-aliasing**: Smooth dot edges for professional quality output
- **Memory Optimization**: Handles large images efficiently
- **Multiple Color Modes**: Black-on-white or white-on-black output
- **High-Quality Resizing**: Maintains aspect ratio with LANCZOS resampling

## Technology Stack

- **Backend**: Django 6.0
- **Image Processing**: OpenCV, PIL/Pillow, NumPy
- **Testing**: Hypothesis (Property-Based Testing)
- **Frontend**: HTML, CSS, JavaScript

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Image Operations
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run migrations:
```bash
python manage.py migrate
```

5. Start the development server:
```bash
python manage.py runserver
```

## Usage

1. Navigate to `http://localhost:8000`
2. Upload an image using the web interface
3. Adjust processing parameters:
   - Dot spacing (10-50 pixels)
   - Dot radius range (1-20 pixels)
   - Color mode (black-on-white or white-on-black)
4. Click "Convert" to generate the dotted version
5. Download the processed image

## Core Algorithm

The image processing pipeline includes:

1. **Image Loading**: PIL-based loading with format normalization
2. **Adaptive Resizing**: Smart resizing based on image characteristics
3. **Grayscale Conversion**: OpenCV-based color space conversion
4. **Gaussian Blur**: Noise reduction for better brightness sampling
5. **Grid Generation**: Adaptive spacing calculation
6. **Dot Rendering**: Gamma-corrected radius scaling with anti-aliasing

## Project Structure

```
Image Operations/
├── converter/              # Main Django app
│   ├── image_processor.py  # Core image processing logic
│   ├── models.py          # Database models
│   ├── views.py           # Web views
│   ├── forms.py           # Django forms
│   └── templates/         # HTML templates
├── image_dotting_converter/ # Django project settings
├── media/                 # User uploads (not in repo)
├── requirements.txt       # Python dependencies
└── manage.py             # Django management script
```

## Testing

The project includes comprehensive property-based tests using Hypothesis:

```bash
python manage.py test
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is open source and available under the MIT License.