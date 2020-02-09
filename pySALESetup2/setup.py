import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pySALESetup",
    version="2.0.0",
    author="James Derrick",
    description="A pre-processing tool for iSALE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jgd10/pySALESetup.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)