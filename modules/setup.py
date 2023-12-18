from setuptools import setup, find_packages

setup(
    name='clusterOVITO',
    version='0.1',
    packages=find_packages(),
    # Additional metadata about your package.
    author='Nick Marcella',
    author_email='nmarcella@bnl.gov',
    description='the protomodule for clusterOVITO',
    install_requires=[
        # Any dependencies your module needs, listed as strings, e.g.,
        # 'requests', 'numpy>=1.13.1', etc.
    ],
)
