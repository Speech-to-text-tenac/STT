# African language Speech Recognition - Speech-to-Text

![AfricanLanguage](./Images/AfricanLangage.png)

This project demonstartes how to build speech-to-text deep learning model that process and convert African language((Amharic/Swahili) in to text.

## Content

- [Introduction](#introduction)
- [Objective](#objective)
- [Data & Features](#data-&-features)
- [Project Structure](#project-structure)
- [contributors](#contributors)
- [Install](#install)

## Introduction

The World Food Program wants to deploy an intelligent form that collects nutritional information of food bought and sold at markets in two different countries in Africa - Ethiopia and Kenya. The design of this intelligent form requires selected people to install an app on their mobile phone, and whenever they buy food, they use their voice to activate the app to register the list of items they just bought in their own language. The intelligent systems in the app are expected to live to transcribe the speech-to-text and organize the information in an easy-to-process way in a database.

Here we will build a deep learning model that is capable of transcribing a speech to text in the Amharic language. The model we produce will be accurate and is robust against background noise.

## Objective

To build a deep learning model that is capable of transcribing a speech to text.

## Data & Features

Dataset for:
[Amharic](https://github.com/getalp/ALFFA_PUBLIC)
[Swahili](https://github.com/getalp/ALFFA_PUBLIC)

Input features (X): audio clips of spoken words  
Target labels (y): a text transcript of what was spoken

## Project Structure

### .dvc

used to track large files, models, dataset directories.

### Images

a directory for images and results

### data

a directory to hold versioned datasets

### notebooks

a directory for notebook files.

### scripts

directory for scripts files.

### root directory

`.dvcignore` : to hide unnecesary files from dvc.  
`.gitignore` :to hide unnecesary at the root directory.  
`LICENSE` : for preservation of copyright and license notices.  
`README.md`: Markdown text with a brief explanation of the project and the repository structure.  
`setup.py`: a configuration file for installing the scripts as a package
`requirements.txt`: a text file lsiting the projet's dependancies

## contributors

![Contributors list](https://contrib.rocks/image?repo=Speech-to-text-tenac/STT)

## Install

```
clone https://github.com/Speech-to-text-tenac/STT
pip install -r requirements.txt
```
