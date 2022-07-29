import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='bright2nuc',
    version='0.0.17',
    description='An easy and convinient nuclei prediction pipline from brightfield in 2D and 3D',
    author='Dominik Waibel',
    author_email='dominik.waibel@helmholtz-muenchen.de',
    license='MIT',
    keywords='Computational Biology Deep Learning',
    #url='https://github.com/marrlab/InstantDL',
    packages=find_packages(exclude=['doc*', 'test*']),
    install_requires=[  'keras>=2.2.4',
                        'tensorboard>=1.13.0'],
    long_description=read('README.md'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
    ],
)
