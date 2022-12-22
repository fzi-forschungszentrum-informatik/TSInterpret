import io
import os
import subprocess
import sys

import setuptools
#subprocess.check_call([sys.executable, "-m", "pip", "install", "Cython"])
#try:
#    from numpy import get_include
#except ImportError:
#    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy ==1.21.6"])
#    from numpy import get_include


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
    "scikit-learn==1.0.2",
    # "scikit-surprise==1.1.1",
    "torch",
    "pandas~=1.3.2",
    "numpy>=1.21.6",
    "tqdm~=4.61.2",
    "h5py==3.7.0", # todo add version
    "joblib>=1.0.1",
    "kaggle==1.5.12",
    "lime==0.2.0.1",
    "Markdown==3.3.4",
    "matplotlib>=3.3.4",
    "partd==1.2.0",
    "pytz>=2021.3",
    "shap>=0.39.0",
    "tensorflow>=2.9.1",
    "keras>=2.9.0",
    "tsfresh>=0.18.0",
    "tslearn>= 0.5.2",
    "seaborn>=0.11.2",
    "scikit_optimize>=0.9.0",
    "mlrose>=1.3.0",
    "torchcam>=0.3.1",
    "tf_explain>=0.3.1",
    "opencv-python==4.6.0.66",
    "captum>= 0.5.0",
    "pyts>=0.12.0",
    "deprecated==1.2.13",
    "pymop",
    "deap", 
    "wheel"
]

dev_packages = base_packages + [
    "flake8>=5.0.4",
    "isort==5.9.3",
    "mypy>=0.761",
    "pre-commit>=2.9.2",
    "pytest>=4.5.0",
    "pytest-cov>=2.6.1",
    "pyupgrade>=3.2.0"
]

docs_packages = [
    "flask>=2.0.2",
    "ipykernel>=6.9.0",
    "mike>=0.5.3",
    "mkdocs>=1.2.3",
    "mkdocs-awesome-pages-plugin>=2.7.0",
    "mkdocs-gen-files>=0.3.5",
    "mkdocs-literate-nav>=0.4.1",
    "mkdocs-material>=8.1.11",
    "mkdocstrings[python]>=0.19.0",
    'pytkdocs[numpy-style]>=0.5.0',
    "ipython_genutils>=0.1.0",
    "mkdocs-jupyter>=0.20.0",
    "mkdocs-bibtex==2.8.1",
    "nbconvert==6.4.2",
    "numpydoc==1.2",
    "spacy==3.2.2",
    "jinja2==3.0.3"
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
        "all": dev_packages + docs_packages,
        ":python_version == '3.8'": ["dataclasses"],
    },
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    ext_modules=[]
)