from setuptools import find_packages, setup

setup(
  # Metadata
  name="aidb",
  version="0.0.1",
  description="",
  long_description=open("README.md").read(),
  long_description_content_type="text/markdown",
  author="Daniel Kang",
  author_email="daniel.d.kang@gmail.com",
  # Packages
  packages=find_packages(include=["aidb", "aidb.*", "aidb_utilities", "aidb_utilities.*"]),
  python_requires=">=3.9",
  classifiers=[
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Information Technology",
    "Intended Audience :: Legal Industry",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: SQL",
    "Programming Language :: Python :: 3 :: Only",
  ],
)
