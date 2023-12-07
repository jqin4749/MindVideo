from setuptools import setup, find_packages

setup(
    name='MindVideo',
    version='0.0.1',
    description='',
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        'torch',
        'numpy',
        'diffuser',
    ],
)