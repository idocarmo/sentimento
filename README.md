# sentimento

**sentimento** spans the activities of the final project of Udacity's Data Science nanodegree program.

In this project we implemented a binary sentiment classifier (positive or negative) from Portuguese imdb reviews dataset [1] using a bidemensional embedding layer.


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
  - datasets
  - ipykernel
  - tensorflow
  - scikit-learn
  - scikit-plot
~~~~~~~~~~~~

### Environment
You can prepare the pyhton environment  with all the dependencies using ``conda``:

    conda env create -f environment.yml

## Repository Content

- 📂 [app](https://github.com/idocarmo/response-pipeline/tree/main/app) contains files used in the deploy ofthe web app;
    - 📂 [templates](https://github.com/idocarmo/response-pipeline/tree/main/app/templates) contains the html templates of the web app.
    - 📄 run.py is the script with the deploying of the web app.
- 📂 [data](https://github.com/idocarmo/response-pipeline/tree/main/data) contains the files used and exported on the data processing step;
    - 📄 disaster_categories.csv is the raw data with categories names for each message
    - 📄 disaster_messages.csv is the raw data with the text messages
    - 📄 disaster_response.db is the SQL database generated when running  *process_data.py*
    - 📄 disaster_response.db is the python script with data cleaning and exporting.
- 📂 [model](https://github.com/idocarmo/response-pipeline/tree/main/model) contains the files used and exported on the classifier model building step;
    - 📄 train_classifier.py is the python script with the model building and training.
    - 📄 trained_classifier.pkl is the trained Random Forest Classifier saved when executing *train_classifier.py*
- 📄 environment.yml is the file with the instructions for python environment building.  

## How to Run

1. Run the following commands in the project's root directory to set up your database and model.

## About the Classifier


## References
[1] Maritaca AI.