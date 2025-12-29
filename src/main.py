import argparse
#allows python code to run terminal commands exactly as if typed into the command prompt
import subprocess
from .data_gathering_cleaning import webscrapper_stats_gathering
from .rumor_gathering import webscrapper_rumors
from .train_binary import run_binary_training
from .train_multiclass import run_multiclass_training
from .batch_prediction_class import batch_prediction
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def run_full_pipeline():
    logging.info("Starting full pipeline: scrape stats + rumors + train + batch preds + dbt seeds")

    logging.info("Step 1/6: Scraping player stats...")
    webscrapper_stats_gathering()

    logging.info("Step 2/6: Scraping rumors...")
    webscrapper_rumors()

    logging.info("Step 3/6: Training binary model...")
    run_binary_training()

    logging.info("Step 4/6: Training multiclass model...")
    run_multiclass_training()

    logging.info("Step 5/6: Running batch binary & multiclass predictions...")
    bp = batch_prediction()
    bp.binary_batch_preds(player_=None)
    bp.multiclass_batch_predict(player_=None)

    logging.info("Step 6/6: Running dbt seeds...")
    #this section allows us to run commands in the command prompt via a python file
    subprocess.run(
        [
            "dbt", "seed",
            "--project-dir", "dbt_league_pred",
            "--profiles-dir", "dbt_league_pred",
        ],
        check=True,
    )

    logging.info("Full pipeline completed successfully.")

def parse_args():
    parser=argparse.ArgumentParser(
        description='Orchestrator for the Transfer Target Model.'
    )

    subparsers = parser.add_subparsers(
        dest='command',
        required=True,
        help='What function do you want to run?'
    )

    #---------------------------- data gathering ---------------------
    subparsers.add_parser(
        'scrape-stats',
        help='Running webscrapper to gather data.'
    )

    subparsers.add_parser(
        'scrape-rumors',
        help='Running webscrapper gather rumors with which league a player is associated.'
    )
    #---------------------------- training commands ---------------------
    subparsers.add_parser(
        'train-binary',
        help='Run binary classifier training (Optuna,calibration and creating final model.)'
    )

    subparsers.add_parser(
        'train-multiclass',
        help='Run multiclass classifier training (Optuna, calibration and creating the final model.)'
    )

    #---------------------------- binary batch prediction ---------------------
    subparsers.add_parser(
        'batch-binary',
        help='Run binary classifier to predict if the player belong in a top or non-top league.'
    )

    subparsers.add_parser(
        'batch-multiclass',
        help='Run multiclass classifier to predict to which league the player '
    )

    subparsers.add_parser(
        'full-run',
        help='This will allow you to run all models/function etc. to run sequential. In other words, it will trigger all function sequential'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    logging.info(f"[DEBUG] command = {args.command}")

    if args.command == "scrape-stats":
        webscrapper_stats_gathering()

    elif args.command == "scrape-rumors":
        webscrapper_rumors()

    elif args.command == "train-binary":
        run_binary_training()

    elif args.command == "train-multiclass":
        run_multiclass_training()

    elif args.command == "batch-binary":
        bp = batch_prediction()
        bp.binary_batch_preds(player_=None)   # all players

    elif args.command == "batch-multiclass":
        bp = batch_prediction()
        bp.multiclass_batch_predict(player_=None)
    elif args.command == "full-run":
        run_full_pipeline()

if __name__ == '__main__':
    main()

#how to call the functions
##python -m src.main scrape-stats
##python -m src.main scrape-rumors
##python -m src.main train-binary
##python -m src.main train-multiclass
##python -m src.main batch-binary
##python -m src.main batch-multiclass
##python -m src.main full-run