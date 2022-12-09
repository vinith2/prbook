---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---


# Installation 

We will be using jupyter book as a long and a verbose form of the lectures slides. This is essentially a collective lecture notes, scribed by the students. The source code for the the book is hosten in the [github repo](https://github.com/swing-research/prbook). 

## Local build
To  run and build the course jupyterbook locally, you need to first install [jupyter-book](https://jupyterbook.org/en/stable/start/overview.html). You can install juptyerbook  using pip with the command:
```
pip install -U jupyter-book
```

If you are using conda, you can install it using:

```
conda install -c conda-forge jupyter-book
```
This is used to build  and generate the html files locally. For more details regarding the installation can check the [link](https://jupyterbook.org/en/stable/start/overview.html). 

We assume you have python and some of the machine learning libraries used in the course installed. Next, you need to download the repository in your local machine. Run the below command to clone the repository in your machine 

```
git clone git@github.com:swing-research/prbook.git
```
The above command download the course jupyterbook files from Github into your local directory. This might take some times as there are several media files to be downloaded. Once downloaded you will have a folder called prbook. The prbook folder should look something like: 

```
./prbook
├── book_data
├── _build
├── _config.yml
├── denoise-deblur.md
├── filtering.md
├── images
├── installation.md
├── introduction.md
├── intro.md
├── linear-methods.md
├── logo.png
├── logo_sada-lab_black.svg
├── modeling-knowledge.md
├── perceptron.py
├── __pycache__
├── references.bib
├── requirements.txt
└── _toc.yml

```

Once downloaded, you can now build prbook to generate the HTML files. Now run the following command: 

```
jupyter-book build prbook
```
This will generate a functional HTML site. The site is placed in  ```./prbook/_build/```. You can open ```index.html``` in the ```/prbook/_build/html/``` folder to view the contents of the book. For more details on building and troubleshooting jupyter-book builds follow the [link](https://jupyterbook.org/en/stable/start/overview.html) 


## Scribing 
This optional task for students to get extra credit or for people who missed the some of the lectures. You will be assigned certain lectures to scribe. Based on the assigned lectures you need to fill the corresponding markup document with python code snippets and explanations pertaining to the lecture. If a file  for the lecture is not present you are free to create a new file with an appropriate name, however it better to give us a prior notification. For reference regarding the style, quality and quantity of explanations for each topic you can look at [Introduction](./introduction.md), [Denoising](./denoise-deblur.html#denoising), [Wiener filter](./denoise-deblur.html#wiener-filter). 

### Editing and Style Convention
The prefered way of editing and creating the lecture notes is through markdown files as done in ```prbook```. But you can create the books using jupyter notebooks as well.  You can look at ```./prbook/introduction.md``` on how to write down the markdown files which contains python snippets from the slides and explanation. Latex macros from ```./prbook/_config.yml```, should be used for equations in the explanation part of the text. The notation scheme used in [Denoising and deblurring](./denoise-deblur.html) should be followed whenever possible. 

### Submitting 
To upload your edits and books, the preferred approach is to submit a pull request (PR), with your changes. Be specific in your commit messages. We will review your changes and based on the quality of work we either ask you to improve on it or accept the changes. If the PR is reviewed and merged into the main branch, the task is considered successful and you obtain full points for the scribing task assigned to you. 

An alternative way to submit your work is through email. Then we evaluate and inform you whether it is acceptable or not. 











