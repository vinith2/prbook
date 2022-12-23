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

We will use jupyter-book as a verbose form of the lecture slides. This is essentially a collective lecture notes, scribed by the students. The source code for the the book is hosted in the [github repo](https://github.com/swing-research/prbook). 

## Local build
To  run and build the course jupyter-book locally, you need to first install [jupyter-book](https://jupyterbook.org/en/stable/start/overview.html). You can install juptyerbook using pip command:
```
pip install -U jupyter-book
```

If you are using conda, you can install it using:

```
conda install -c conda-forge jupyter-book
```
For more details regarding the installation  you can check the [link](https://jupyterbook.org/en/stable/start/overview.html). 

We assume you have python and some of the machine learning libraries used in the course installed. Next, you need to download the repository in your local machine. Run the below command to clone the repository in your machine 

```
git clone git@github.com:swing-research/prbook.git
```
The above command downloads the course jupyter-book files from Github into your local directory. This might take some times as there are several media files to be downloaded. Once downloaded you will have a folder called prbook. The prbook folder should look something like: 

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

You are ready to build prbook and generate the HTML files. Now run the following command: 

```
jupyter-book build prbook
```
This will generate a functional HTML site. The site is placed in  ```./prbook/_build/```. You can open ```index.html``` in the directory ```/prbook/_build/html/``` folder to view the contents of the book. For more details on building and troubleshooting jupyter-book builds follow the [link](https://jupyterbook.org/en/stable/start/overview.html) 


## Scribing 
This is an optional task for students to get extra credit or for people who missed some lectures with attendance. You will be assigned certain lectures to scribe. Based on the assigned lectures you need to fill the corresponding Markdown document with python code snippets and explanations pertaining to the lecture. If a file  for the lecture is not present you are free to create a new file with an appropriate name, however it is better to give us a prior notification. Next, you need to add the name of the file in ```_toc.yml``` in the appropriate order.  For reference regarding the style and quality of explanations for each topic you can look at [Introduction](./introduction.md), [Denoising](./denoise-deblur.html#denoising), [Wiener filter](./denoise-deblur.html#wiener-filter). 

### Editing and Style Convention
The prefered way of editing and creating the lecture notes is through Markdown files as done in ```prbook```. But you can create the books using jupyter notebooks as well.  You can look at ```./prbook/introduction.md``` on how to write down the markdown files which contains python snippets from the slides and explanation. Latex macros from ```./prbook/_config.yml```, should be used for equations in the explanation part of the text. Use Latex for equations and math terms. Follow the notation scheme in  [Denoising and deblurring](./denoise-deblur.html) whenever possible. 

### Submitting 
To upload your work, the preferred approach is to submit it as a pull request (PR), with your changes. An example PR is given [below](#pull-request-Example ) Be specific in your commit messages. We will review your changes and based on the quality of work we either ask you to improve on it or accept the changes. If the PR is reviewed and merged into the main branch, the task is considered successful and you obtain full points for the task assigned to you. 

An alternative way to submit your work is through email. Then we evaluate and inform you whether it is acceptable or not. 


### Pull request Example

Here we look at an example pull request to change a word in [Installation](./installation.md). First you open the [github repo](https://github.com/swing-research/prbook) in your browser. You need to have a github account to submit your changes. Once in prbook gihtub page, you need to fork it (Click on the top right button with the name ```Fork```). This makes a copy of the repository in your account. Once forked, you need to clone the personal copy of the reposity into your local machine. The clone command looks like:
```
git clone git@github.com:< your github username >/prbook.git
```
You will have the personal copy of ```prbook``` downloaded in your local machine.  Open the file ```./prbook/installation.md``` and go to line '86' and change the text from '<your github username>' to '<username>' or you can change it your github user name as well. Once changed, you need push the local changes to the github server. You can run the following command to push the changes. 


```
git add ./instruction.md 
git commit -m 'changing the user name in line 89' 
git push
```

**NOTE:**
Be specific in your commit messages, this helps in debugging if there are any errors.

The above command uploads the local changes to your copy of prbook in the github server. Now, you need to go to your github page and go to your copy of the prbook. The hyperlink to your copy should look like ```https://github.com/<username>/prbook```.  You can see the changes in the browser as well. Now you are read to push the changes to the main jupyter-book. On the browser you can see a button called ```contribute```. Clicking on this button, gives you an option to open a pull request. When you click this, you are in the 'Open a pull request' page.  You get an option to add title and write comments about your changes. Once done click on the ```Create pull request``` button.  Now your pull request is submmitted. A reviewer will be assigned and you can look at the comments regarding the your updates in the [pull request tab](https://github.com/swing-research/prbook/pulls). 













