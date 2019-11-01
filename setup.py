import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dlutils",
    version="0.0.1",
    author="Jean Ollion",
    author_email="jean.ollion@polytechnique.edu",
    description="",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/jeanollion/dlutils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
    install_requires=['h5py>=2.9,==1.14.0', 'numpy', 'scipy', 'scikit-learn', 'keras']
)
