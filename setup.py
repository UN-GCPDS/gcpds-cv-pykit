#!/usr/bin/env python3
"""
Setup script for GCPDS Computer Vision Python Kit
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure we're in the right directory
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

def read_requirements(filename='requirements.txt'):
    """Read requirements from requirements.txt file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return requirements
    except FileNotFoundError:
        # Fallback requirements if requirements.txt is not found
        return [
            'torch>=2.0.0',
            'torchvision>=0.15.0',
            'numpy>=1.21.0',
            'tqdm>=4.64.0',
            'matplotlib>=3.5.0',
            'wandb>=0.15.0',
            'opencv-python>=4.6.0',
            'Pillow>=9.0.0',
            'scipy>=1.9.0',
            'pandas>=1.4.0',
        ]

def get_version():
    """Get version from package"""
    version_file = os.path.join('gcpds_cv_pykit', '_version.py')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            exec(f.read())
            return locals()['__version__']
    return '0.1.0'  # Default version

# Read requirements
install_requires = read_requirements()

# Development dependencies
dev_requires = [
    'pytest>=7.0.0',
    'pytest-cov>=4.0.0',
    'black>=22.0.0',
    'flake8>=5.0.0',
    'isort>=5.10.0',
    'pre-commit>=2.20.0',
]

# Documentation dependencies
docs_requires = [
    'sphinx>=5.0.0',
    'sphinx-rtd-theme>=1.0.0',
    'sphinx-autodoc-typehints>=1.19.0',
    'myst-parser>=0.18.0',
]

# Jupyter/notebook dependencies
jupyter_requires = [
    'jupyter>=1.0.0',
    'ipywidgets>=8.0.0',
    'notebook>=6.4.0',
]

# All extra dependencies
all_requires = dev_requires + docs_requires + jupyter_requires

setup(
    name='gcpds-cv-pykit',
    version=get_version(),
    author='GCPDS Team',
    author_email='gcpds_man@unal.edu.co',
    description='A comprehensive toolkit for computer vision and segmentation tasks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/UN-GCPDS/gcpds-cv-pykit',
    project_urls={
        'Bug Reports': 'https://github.com/UN-GCPDS/gcpds-cv-pykit/issues',
        'Source': 'https://github.com/UN-GCPDS/gcpds-cv-pykit',
        'Documentation': 'https://gcpds-cv-pykit.readthedocs.io/',
    },
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require={
        'dev': dev_requires,
        'docs': docs_requires,
        'jupyter': jupyter_requires,
        'all': all_requires,
    },
    entry_points={
        'console_scripts': [
            'gcpds-train=gcpds_cv_pykit.cli.train:main',
            'gcpds-evaluate=gcpds_cv_pykit.cli.evaluate:main',
        ],
    },
    package_data={
        'gcpds_cv_pykit': [
            'configs/*.yaml',
            'configs/*.yml',
            'configs/*.json',
            'data/*.json',
            'data/*.yaml',
            'data/*.yml',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        'computer vision',
        'segmentation',
        'deep learning',
        'pytorch',
        'unet',
        'medical imaging',
        'image processing',
        'machine learning',
        'artificial intelligence',
    ],
    license='MIT',
    platforms=['any'],
    
    # Additional metadata
    maintainer='GCPDS Team',
    maintainer_email='gcpds_man@unal.edu.co',
    
    # Ensure compatibility
    setup_requires=[
        'setuptools>=45',
        'wheel',
    ],
    
    # Test configuration
    test_suite='tests',
    tests_require=[
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
    ],
    
    # Options for different installation scenarios
    options={
        'build_scripts': {
            'executable': '/usr/bin/python3',
        },
    },
)

# Post-installation message
if __name__ == '__main__':
    print("\n" + "="*60)
    print("GCPDS Computer Vision Python Kit Installation Complete!")
    print("="*60)
    print("Thank you for installing gcpds-cv-pykit!")
    print("\nQuick start:")
    print("  from gcpds_cv_pykit.baseline.trainers import SegmentationModel_Trainer")
    print("  from gcpds_cv_pykit.baseline import PerformanceModels")
    print("\nDocumentation: https://gcpds-cv-pykit.readthedocs.io/")
    print("Issues: https://github.com/UN-GCPDS/gcpds-cv-pykit/issues")
    print("="*60 + "\n")