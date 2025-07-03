from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

    setup(
    name='gmfold_pipeline',
    version='0.1',
    description='Pipeline for running GMfold on DNA/RNA SELEX data.',
    author='Justin Baker',
    author_email='baker@math.utah.edu',
    package_dir={'': 'src'},  # Source directory
    packages=find_packages(where='src'),  # Automatically find packages in src
    include_package_data=True,
    )
