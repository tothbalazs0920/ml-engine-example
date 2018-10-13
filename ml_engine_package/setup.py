from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow == 1.5.0', 'numpy == 1.14.0', 'Keras == 2.1.5', 'h5py == 2.7.1', 'pandas==0.22.0']

setup(
    name='newsgroupclassification',
    version='0.1',
    author='',
    author_email='',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='',
    requires=[]
)
