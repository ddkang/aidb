import os

from setuptools import find_packages, setup

from aidb_utilities.extra_dependencies_handler import (
    EXTRA_DEPENDENCIES_MAPPING, get_extra_dependencies)

with open('requirements.txt') as f:
  required = f.read().splitlines()

setup(
  # Metadata
  name="ai-db",
  version="0.0.2",
  description="Analyze your unstructured data",
  long_description=open("README.md").read(),
  long_description_content_type="text/markdown",
  author="Daniel Kang",
  author_email="daniel.d.kang@gmail.com",
  # Packages
  packages=find_packages(include=["aidb", "aidb.*", "aidb_utilities", "aidb_utilities.*"]),
  python_requires=">=3.9",
  install_requires=required,
  extras_require={
      name: get_extra_dependencies(mapping, name)
      for name, mapping in EXTRA_DEPENDENCIES_MAPPING.items()
  },
  classifiers=[
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: SQL",
    "Programming Language :: Python :: 3 :: Only",
  ],
)
