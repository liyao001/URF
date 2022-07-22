from setuptools import setup
from codecs import open
from os import path
import re

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


def read(*parts):
    with open(path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='URF',
    version=find_version("URF", "__init__.py"),
    install_requires=['scikit-learn>=0.17', 'numpy>=1.9.0', 'scipy', 'pycluster'],
    url='https://github.com/liyao001/URF',
    packages=["URF", ],
    install_requires=["scikit-learn", ],
    license='Apache 2.0',
    author='Li Yao',
    author_email='yaol17@mails.tsinghua.edu.cn',
    description='Unsupervised Random Forest (Random Forest Clustering)',
    long_description=long_description,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering'
    ],
)
