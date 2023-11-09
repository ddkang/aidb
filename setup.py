from setuptools import find_packages, setup

setup(
  # Metadata
  name="aidb",
  version="0.0.1",
  description="Analyze your unstructured data",
  # long_description=open("README.md").read(),
  # long_description_content_type="text/markdown",
  author="Daniel Kang",
  author_email="daniel.d.kang@gmail.com",
  # Packages
  packages=find_packages(include=["aidb", "aidb.*", "aidb_utilities", "aidb_utilities.*"]),
  python_requires=">=3.9",
  classifiers=[
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache License",
    "Operating System :: OS Independent",
    "Programming Language :: SQL",
    "Programming Language :: Python :: 3 :: Only",
  ],
)
