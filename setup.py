from setuptools import find_packages, setup

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
  classifiers=[
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: SQL",
    "Programming Language :: Python :: 3 :: Only",
  ],
)
