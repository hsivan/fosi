import setuptools

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

with open('requirements.txt') as fid:
    INSTALL_REQUIRES = [line.strip() for line in fid.readlines() if line]

# get __version__ from fosi/version.py
_dct = {}
with open('fosi/version.py') as f:
    exec(f.read(), _dct)
VERSION = _dct['__version__']

setuptools.setup(
    name='fosi',
    version=VERSION,
    license='3-Clause BSD',
    author="Hadar Sivan",
    author_email="hadarsivan@cs.technion.ac.il",
    description="FOSI library for hybrid first and second order optimization.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    install_requires=INSTALL_REQUIRES,
    url='https://github.com/hsivan/fosi',
    packages=setuptools.find_packages(include=['fosi', 'fosi.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
