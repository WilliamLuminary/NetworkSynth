from setuptools import setup, find_packages
import os

# Read the requirements from the requirements.txt file
def read_requirements():
    with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as req_file:
        return req_file.read().splitlines()

setup(
    name='NetworkSynth',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements(),  # Read from requirements.txt
    entry_points={
        'console_scripts': [
            'networksynth = src.main:main',  # Example entry point for your main script
        ],
    },
)