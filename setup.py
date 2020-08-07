import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FRETlines",
    version="0.0.4",
    author="Anders Barth",
    author_email="anders.barth@gmail.com",
    description="Package for the calculation of FRET-lines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Fluorescence-Tools/FRETlines",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)