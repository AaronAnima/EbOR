import sys
from setuptools import setup, find_packages
from ipdb import set_trace

assert sys.version_info.major == 3 and sys.version_info.minor >= 9, \
    "WMD uses Python 3.6 and above. "

with open('README.md', 'r') as f:
    # description from readme file
    long_description = f.read()

setup(
    name='ebor',
    version='0.1',
    author='Mingdong Wu',
    author_email='wmingd@pku.edu.cn',
    description='Environments for Example-based Object Rearrangement.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    license='MIT license',
    url='https://github.com/AaronAnima/EbOR',
    install_requires=[
        'gym>=0.20.0,<0.25.0a0',
        'numpy',
        'pybullet>=3.2.5',
    ],
    python_requires='>=3.6',
    platforms=['Linux Ubuntu'],  
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)