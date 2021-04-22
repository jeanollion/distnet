import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DiST_Net",
    version="0.1.2",
    author="Jean Ollion",
    author_email="jean.ollion@polytechnique.org",
    description="keras implementation of DiSTNet & utilities for mother machine data analysis with keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeanollion/distnet",
    download_url = 'https://github.com/jeanollion/distnet/archive/v_01.tar.gz',
    packages=setuptools.find_packages(),
    keywords = ['Segmentation', 'Tracking', 'SelfAttention', 'Tensorflow', 'Keras'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3',
    install_requires=['numpy', 'scipy', 'tensorflow', 'keras_preprocessing', 'edt==2.0.2', 'dataset_iterator>=0.2.1']
)
