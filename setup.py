import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="distnet",
    version="0.1",
    author="Jean Ollion",
    author_email="jean.ollion@polytechnique.org",
    description="keras implementation of DiSTNet & utilities for mother machine data analysis with keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeanollion/distnet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
    install_requires=['numpy', 'scipy', 'tensorflow', 'keras_preprocessing', 'edt', "dataset_iterator"]
)
