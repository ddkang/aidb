import os

from setuptools import find_packages, setup

from aidb_utilities.requirements_parser import get_main_and_extras

main_requirements, extras_require = get_main_and_extras()

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
  install_requires=main_requirements,
  extras_require=extras_require,
  classifiers=[
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: SQL",
    "Programming Language :: Python :: 3 :: Only",
  ],
)
