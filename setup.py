from setuptools import setup, find_packages

setup(
    name='action_recognition',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'opencv-python',
        'scikit-learn',
        'keras',
        'tensorflow'
    ],
    entry_points={
        'console_scripts': [
            'run-main=src.main:main',
        ],
    },
)