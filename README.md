# sentimento

**sentimento** spans the activities of the final project of Udacity's Data Science nanodegree program.

In this project we implemented a binary sentiment classifier (positive or negative) from Portuguese imdb reviews dataset [1] using a bidemensional embedding layer.

For further model discussion please check the medium article [Text Classification with Embeddings]()


## Source code
You can clone this repository in your machine with the command:

    git clone https://github.com/idocarmo/sentimento.git

## Project Setup
### Dependencies

response-pipeline project requires:
~~~~~~~~~~~~
  - python=3.8
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - datasets
  - tensorflow
~~~~~~~~~~~~

### Environment
You can prepare the pyhton environment  with all the dependencies using ``conda``:

    conda env create -f environment.yml

Alternatively you can use your local python 3 instalation and run

    pip install -r requirements.txt

## Repository Content

- 📂 [data](https://github.com/idocarmo/response-pipeline/tree/main/app) nothing there but I like to have it in my repos 😳;
- 📂 [model](https://github.com/idocarmo/response-pipeline/tree/main/app) contains training logs;
- 📂 [notebooks](https://github.com/idocarmo/response-pipeline/tree/main/data) contains notebooks used in the development;
    - 📄 1.0-icc-eda.ipynb is the jupyter notebook with the exploratory data analysis.
- 📂 [reports](https://github.com/idocarmo/response-pipeline/tree/main/model) contains the results reports of the model;
- 📂 [src](https://github.com/idocarmo/response-pipeline/tree/main/model) contains source code files used to buil the model;
    - 📄 data.py is the script used for data processing.
    - 📄 evaluation.py is the script with some useful functions for consolidating the model results.
    - 📄 model_training.py is the script where we build and train our sentiment classification model.
- 📄 environment.yml is the file with the instructions for buildign the python environment.
- 📄 requirements.txt is the alternative file with the instructions for building the python environment.   


## References
[1] [IMDB Reviews Portuguese Dataset](https://medium.com/r/?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Fmaritaca-ai%2Fimdb_pt), Maritaca-AI.
