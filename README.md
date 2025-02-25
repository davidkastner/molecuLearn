![Graphical Summary of README](docs/_static/header.webp)
molecuLearn
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/davidkastner/molecuLearn/workflows/CI/badge.svg)](https://github.com/davidkastner/molecuLearn/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/moleculearn/badge/?version=latest)](https://moleculearn.readthedocs.io/en/latest/?badge=latest)


## Table of Contents
1. **Overview**
    * Introduction
2. **Quick Start**
    * Accessing QA functions
3. **Installation**
    * Installing molecuLearn
    * Prerequisites
4. **What is included?**
    * File structure
5. **Documentation**
    * Read the Docs
    * Examples
6. **Developer Guide**
    * GitHub refresher


## 1. Overview
The objective of molecuLearn (mL), is to facilitate the application of machine learning to extract patterns from ab-initio molecular dynamics simulations.


## 2. Quick start
![Welcome screen help options](docs/_static/welcome_help_demo.png)

To get started, once `ml` has been installed, run `ml --help` or `ml -h` to see the available actions.

## 3. Installation
Install the package by running the follow commands inside the repository. This will perform a developmental version install. It is good practice to do this inside of a virtual environment. A yaml environmental file has been created to automate the installation of dependencies.

### Setup developing environment
To begin working with molecuLearn, first clone the repo and then move into the top-level directory of the package.
Then perform a developer install.
Remember to update your GitHub [ssh keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).
```bash
git clone git@github.com:davidkastner/molecuLearn.git
cd molecuLearn
```

### Creating python environment
All the dependencies can be loaded together using the prebuilt environment.yml file.
Compatibility is automatically tested for python versions 3.8 and higher.
If you are only going to be using the package run:
```bash
conda env create -f environment.yml
conda activate ml # Alternatively you may need source activate ml
python -m pip install -e .
```

### Command-line interface
All of the functionality of molecuLearn has been organized into a command-line interface (CLI).
With one additional step, the CLI can be called from anywhere.
We just have to setup a shortcut command in your BASHRC.
Add the following line to your BASHRC:
```bash
alias ml='python /path/to/the/molecuLearn/cli.py
```


## 4. What is included?
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


## 5. Documentation
### Run the following commands to update the ReadTheDocs site
```bash
make clean
make html
```


## 6. Developer guide
### GitHub refresher for those who would like to contribute
#### Push new changes
```bash
git status
git pull
git add -A .
git commit -m "Change a specific functionality"
git push -u origin main
```

#### Making a pull request
```
git checkout main
git pull
# Before you begin making changes, create a new branch
git checkout -b new-feature-branch
git add -A
git commit -m "Detailed commit message describing the changes"
git push -u origin new-feature-branch
# Visit github.com to add description, submit, merge the pull request, and delete the remote branch
# Once finished on github.com, return to local
git checkout main
git pull
git branch -d new-feature-branch
```

#### Handle merge conflict

```
git stash push --include-untracked
git stash drop
git pull
```

### Copyright

Copyright (c) 2023, David W. Kastner


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
