# Transfer Target Model

The following project is supposed to predict, based on a players last season stats (season 2024-2025) which league the player will join. Two models have been trained first a binary model to see if the player will join a top or a non top-league (binary model). If a player is more likely to join a top league, than the multiclass model will be used to predict if a player will join the Bundesliga, Ligue 1, Premier League, Serie A or LaLiga.
How will it be determined, which league or output is more likely? These models will operate on probability, the class with the highest probability will be the final predicted class.
The binary model will/was be tuned and trained with brier-score, the multiclass model will/was trained on log-loss.
The data was gathered from fbref.com, fbref provides the player statistic from the seasons '2017-2018' - '2024-2025'. From Transfermarket.com, the transfer date and fee was gathered, addtionally the number of rumors were gathered for each player

## DataSources
The folder "DataSources" contains different datasets:
- Model_Data => it contains the test, train and whole data for the binary and multiclass model. These datasets will be created during the training process and will be saved, when the final model is created
- Prediction => in this folder-substructure will be the predicted values stored, when manually a list of predictions will be triggered. In this folder, in the csv file will contain these values
- Processed => in here all gathered data from the webscrappers are stored, but also will be used for further transformations and the data in there will be used for training, evaluation and predictions
- Raw => in here the raw data of the different leagues is stored. This data will be used for further transformations, so it can be used for training

## dbt_league_pred
In this subfolder, everything is setup for db-runs to create a database and store the total prediction. In the "seeds" subfolder, the total predictions will be stored. What do I mean by total predictions? In the folder "src" are all the python files stored that are needed to gather data, tune/train the model and to make predictions. In this folder, the file "batch_prediction_class" allows us to predict for all available players to predict the if the players belong to a top or non top-league. If a player belongs to a top-league, the player's stats will be passed to the multiclass model, to see which league is more likely to be his new destinantion.
These CSV files will be saved in the "seeds", from this subfolder, it can be dbt-runs triggered and be stored in a database (DuckDatabase).
How to run dbt:
- dbt seed
- dbt build

## Models
This is the location of the final model;
- Final_Encoder: This is the encoder used for the multiclass model
- Final_Model: It is devided into "Binary" and "Multiclass". Each folder contains the final model, the best params from the tuning, columns needed and even a translation of the multiclass. This will be optained during the tuning and training processs. This can be found in the "src" folder, the files "train_binary" and "train_multiclass" train the models and save them in the "Models" folder.

## Notebook
The notebook folder contains following subfolders
- Data Cleaning
    - This subfolder contains the notebook which cleans the aggregated data. It adds the column to disdinguish between top and non-top league.
- DataGathering
    - This subfolder contains the nootbook to run the fbref webscrapper, transfermarket webscrapper for the transfer fee and rumors and combining the raw data
- Experimnts
    - The notebook to create the different drift reports
        - Data (Raw & Transformed)
        - Prediction Drift (Label, Entrop, Max Confidence and Margin)
- Exploratory
    - This subfolder holds a notebook which provides an overview of the data's characteristic so the best suiting model can be selected for the tuning & training.
- Model Deployment
    - Setting up within notebook predictions, API (Postman API) and setting up the prediction class, which will be used later to make the predictions.
- Model_Training_&_Evaluation
    - In here are the notebooks stored for the training, tuning and finalization of the best model. One notebook which trained and tuned the models without the rumor columns, the other notebook, trains and tunes on data, which contain rumors. Here it is observable how the model is evaluated and finalized.

## src
- This folder holds all python files, which will be used in the docker file and is supposed to facilitate a smooth operation, when it is integrated into a pipeline.
    - __init__.py => allows all python files to be turned into a package, so the function and classes can be called in a docker file etc.
    - api_connection.py => this file will be used, to set up the server so it can access the models via api postman => it is called as following => python -m src.api_connection
    - batch_prediction_class.py => this file contains the batch prediction class, this class contains multipel functions, which provide an overview of the most recent stats, manuell single value predictions (binary & multiclass), manuell mutliple value prediction (binary & multiclass), default setting to predict all available player for binary and multiclass.
    - Checking_for_data_drift.py => python file to check for drift in the data and prediction. Same logic as in the notebook in the "Experiments" folder. The file is called as following  => python -m src.Checking_for_data_drift.
    - cleaning.py => provides final aggregation and ML/DL ready. The file is called as following: python -m src.Checking_for_data_drift.
    - data_gathering_cleaning.py => this files gathers the player statistic from the fbref website => it is called with python -m src.data_gathering_cleaning.
    - exploratory.py => this file returns the exploratory analysis in the terminal. python -m src.exploratory
    - main.py => the main file allows to set up a sort of pipeline, which can be used in a docker container or can be passed into a CI/CD. The file is build up as following;
        - First the def function "run_full_pipeline" calling all the needed classes and functions. At the same time, we load the dbt commands, which can be executed in the command prompt via the main file.
        - Next, it sets up functions so they can be called separtly in the command prompt or in a pipeline
            - the functions can be called as following;
                - python -m src.main scrape-stats
                - python -m src.main scrape-rumors
                - python -m src.main train-binary
                - python -m src.main train-multiclass
                - python -m src.main batch-binary
                - python -m src.main batch-multiclass
                - python -m src.main full-run
    - PY_Class_Def.py => this file holds all def-function and classes created for this project. There is another "PY_Class_Def.py" which just imports all def-functions and classes again so we can use it in the docker container.
    - rumor_gathering.py => this file gathers data from Transfermarket.com which associated with the players that the stats and transfer fees have been already gathered. The file is called with "python -m src.rumor_gathering"
    - train_binary.py => this file allows to train & tune the binary model. Note, this file used the best model from the training and tuning notebook. The model might needs to be chnaged, when new data is added. This file will also be triggered in the docker container, to run the file separately => "python -m src.train_binary"
    - train_multiclass.py => this file allows to train & tune the multiclass model. Note, this file used the best model from the training and tuning notebook. The model might needs to be chnaged, when new data is added. This file will also be triggered in the docker container, to run the file separately => "python -m src.train_multiclass"
    - transfer_target_data_cleaning.py => this file holds specific def-function to sort and clean existing transfer data.
    - web_api.py => this file is needed, to facilitate the deployment via streamlit. By creating the website the models can be accessed via a dropdown. The file is ran as following "streamlit run src/web_api.py"

## Dockerfile
This docker container is supposed to be integrated into pipelines, but should also facilitate a smooth operation from a local machine, where a files/scripts are sequentially triggered.
- Building the docker image => docker build -t transfer-target .
- How to run the container on docker image => docker run --rm transfer-target

## requirements
This text file stores all needed libaries and versions, that were used for the project. The file will be installed as following: "pip install -r requirements.txt"

## Quickstart
Here a quick guide to use the models and files the most efficent way
1. docker build -t transfer-target . => building the docker image
2. docker run --rm transfer-target => running the full pipeline, this can also be done with "python -m src.main full-run"
3. Run Individual Pipeline Steps (Optional)
    docker run --rm transfer-target python -m src.main scrape-stats
    docker run --rm transfer-target python -m src.main scrape-rumors
    docker run --rm transfer-target python -m src.main train-binary
    docker run --rm transfer-target python -m src.main train-multiclass
    docker run --rm transfer-target python -m src.main batch-binary
    docker run --rm transfer-target python -m src.main batch-multiclass