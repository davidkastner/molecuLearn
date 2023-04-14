![Graphical Summary of README](docs/_static/header.webp)
molecuLearn
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/davidkastner/molecuLearn/workflows/CI/badge.svg)](https://github.com/davidkastner/molecuLearn/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/moleculearn/badge/?version=latest)](https://moleculearn.readthedocs.io/en/latest/?badge=latest)


## Table of Contents
1. **Overview**
    * Introduction
    * Purpose
2. **Installation**
    * Installing quantumAllostery
    * Prerequisites
3. **What is included?**
    * File structure
4. **Documentation**
    * Read the Docs
    * Examples
5. **Developer Guide**
    * GitHub refresher


## 1. Overview
The objective of molecuLearn (mL), is to facilitate the application of machine learning to extract patterns from ab-initio molecular dynamics simulations.


## 2. Installation
Install the package by running the follow commands inside the repository. This will perform a developmental version install. It is good practice to do this inside of a virtual environment. A yaml environmental file has been created to automate the installation of dependencies.

### Creating python environment
All the dependencies can be loaded together using the prebuilt environment.yml file.
Compatibility is automatically tested for python versions 3.8 and higher.
If you are only going to be using the package run:
```bash
conda env create -f environment.yml
source activate ml
```

### Setup developing environment
To begin working with molecuLearn, first clone the repo and then move into the top-level directory of the package.
The perform a developer install.
Remember to update your GitHub [ssh keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).
```bash
git clone git@github.com:davidkastner/molecuLearn.git
cd molecuLearn
python -m pip install -e .
```

### Command-line interface
All of the functionality of molecuLearn has been organized into a command-line interface (CLI).
With one additional step, the CLI can be called from anywhere.
We just have to setup a shortcut command in your BASHRC.
Add the following line to your BASHRC:
```bash
alias ml='python /path/to/the/molecuLean/cli.py
```


## 3. What is included?
### File structure
```
.
|── cli.py          # Command-line interface entry point
├── docs            # Readthedocs documentation site
├── ml              # Directory containing the quantumAllostery modules
│   ├── process     # Processes raw dynamics data
│   ├── predict     # Machine learning analysis
│   ├── manage      # File management functionality and routines
│   ├── analyze     # Data analysis to combine process and plot routines
│   └── plot        # Automated plotting and vizualization 
└── ...
```


## 4. Documentation
### Run the following commands to update the ReadTheDocs site
```bash
make clean
make html
```


## 5. Developer guide
### GitHub refresher for those who would like to contribute
#### Push new changes
```bash
git status
git pull
git add .
git commit -m "Change a specific functionality"
git push -u origin main
```


### Copyright

Copyright (c) 2023, David W. Kastner


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
