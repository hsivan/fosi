import setuptools

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

# get __version__ from fosi/version.py
_dct = {}
with open('fosi/version.py') as f:
    exec(f.read(), _dct)
VERSION = _dct['__version__']

setuptools.setup(
    name='fosi',
    version=VERSION,
    license='Apache 2.0',
    author="Hadar Sivan",
    author_email="hadarsivan@cs.technion.ac.il",
    description="FOSI library for hybrid first and second order optimization.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url='https://github.com/hsivan/fosi',
    packages=setuptools.find_packages(include=['fosi']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
