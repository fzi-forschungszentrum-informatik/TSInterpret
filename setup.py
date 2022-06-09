import io
import os
import subprocess
import sys

import setuptools

try:
    from numpy import get_include
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    from numpy import get_include

# Package meta-data.
NAME = "TSInterpret"
DESCRIPTION = "todo"
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://ipe-wim-gitlab.fzi.de/hoellig/interpretabilitytimeseries"
EMAIL = "hoellig@fzi.de"
AUTHOR = "Jacqueline Hoellig"
REQUIRES_PYTHON = ">=3.6.0"

# Package requirements.
base_packages = [
    "absl-py==0.14.1",
    "astunparse==1.6.3",
    "cached-property==1.5.2",
    "cachetools==4.2.4",
    "certifi==2021.5.30",
    "charset-normalizer==2.0.6",
    "clang==5.0",
    "click==8.0.3",
    "cloudpickle==2.0.0",
    "cycler==0.10.0",
    "dask==2021.9.1",
    "decorator==4.4.2",
    "distributed==2021.9.1",
    "flatbuffers==1.12",
    "fsspec==2021.10.1",
    "gast==0.4.0",
    "google-auth==1.35.0",
    "google-auth-oauthlib==0.4.6",
    "google-pasta==0.2.0",
    "grpcio==1.41.0",
    "h5py==3.1.0",
    "HeapDict==1.0.1",
    "idna==3.2",
    "imageio==2.9.0",
    "importlib-metadata==4.8.1",
    "Jinja2==3.0.2",
    "joblib==1.0.1",
    "kaggle==1.5.12",
    "keras==2.6.0",
    "Keras-Preprocessing==1.1.2",
    "kiwisolver==1.3.1",
    "lime==0.2.0.1",
    "llvmlite==0.36.0",
    "locket==0.2.1",
    "Markdown==3.3.4",
    "MarkupSafe==2.0.1",
    "matplotlib==3.3.4",
    "matrixprofile==1.1.10",
    "msgpack==1.0.2",
    "networkx==2.5.1",
    "numba==0.53.1",
    "numpy==1.19.5",
    "oauthlib==3.1.1",
    "opt-einsum==3.3.0",
    "packaging==21.0",
    "pandas==1.1.5",
    "partd==1.2.0",
    "patsy==0.5.2",
    "Pillow==8.3.2",
    "protobuf==3.11.2",
    "psutil==5.8.0",
    "pyasn1==0.4.8",
    "pyasn1-modules==0.2.8",
    "pyparsing==2.4.7",
    "python-dateutil==2.8.2",
    "python-slugify==5.0.2",
    "pytz==2021.3",
    "PyWavelets==1.1.1",
    "PyYAML==6.0",
    "requests==2.26.0",
    "requests-oauthlib==1.3.0",
    "rsa==4.7.2",
    "scikit-image==0.17.2",
    "scikit-learn==0.24.2",
    "scipy==1.5.4",
    "shap==0.39.0",
    "Shapely==1.7.1",
    "six==1.15.0",
    "sklearn==0.0",
    "sktime==0.8.0",
    "slicer==0.0.7",
    "sortedcontainers==2.4.0",
    "statsmodels==0.12.2",
    "stumpy==1.9.2",
    "tblib==1.7.0",
    "tensorboard==2.6.0",
    "tensorboard-data-server==0.6.1",
    "tensorboard-plugin-wit==1.8.0",
    "tensorflow==2.6.0",
    "tensorflow-estimator==2.6.0",
    "termcolor==1.1.0",
    "text-unidecode==1.3",
    "threadpoolctl==3.0.0",
    "tifffile==2020.9.3",
    "toolz==0.11.1",
    "tornado==6.1",
    "tqdm==4.62.3",
    "tsfresh==0.18.0",
    "typing-extensions==3.7.4.3",
    "urllib3==1.26.7",
    "Werkzeug==2.0.2",
    "wildboar==1.0.10",
    "wrapt==1.12.1",
    "zict==2.0.0",
    "zipp==3.6.0",

]

dev_packages = base_packages + [
    "graphviz>=0.10.1",
    "matplotlib>=3.0.2",
    "mypy>=0.761",
    "pre-commit>=2.9.2",
    "pytest>=4.5.0",
    "pytest-cov>=2.6.1",
    "scikit-learn>=0.22.1",
    "sqlalchemy>=1.4",
]

docs_packages = [
    "flask==2.0.2",
    "ipykernel==6.9.0",
    "mike==0.5.3",
    "mkdocs==1.2.3",
    "mkdocs-awesome-pages-plugin==2.7.0",
    "mkdocs-material==8.1.11",
    "mkdocstrings==0.18.0",
    "ipython_genutils==0.1.0",
    "mkdocs-jupyter==0.20.0",
    "nbconvert==6.4.2",
    "numpydoc==1.2",
    "spacy==3.2.2",
    "jinja2==3.0.3",
]

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)

# Where the magic happens:
setuptools.setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=base_packages,
    extras_require={
        "dev": dev_packages,
        "test": dev_packages,
        "docs": docs_packages,
        "all": dev_packages,# + docs_packages,
        ":python_version == '3.6'": ["dataclasses"],
    },
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    ext_modules=[]
)
