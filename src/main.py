# Main code for experiment adapted from:
# Ding et al. (2020), "CogLTX: Applying BERT to Long Texts"
# https://github.com/Sleepychord/CogLTX
# License: MIT

import json
import pandas as pd
import torch.nn.functional as F

from pathlib import Path
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from external.cogltx.judge_module import JudgeModule
from external.cogltx.reasoner_module import ReasonerModule
from src.train import train_model
from src.predict import prediction

root_dir = Path.cwd()
output_dir = root_dir / 'outputs'
data_dir = root_dir / 'data' / 'testing'
result_dir = root_dir / 'results'/ 'final'

# Set up argument parser
def set_parser_arguments(parser):
    # Arguments for output directories
    parser.add_argument("--save_dir", type=str, default='save_dir', help="directory to save models")
    parser.add_argument("--tmp_dir", type=str, default='tmp_dir', help="directory to save temporary ddp files")
    parser.add_argument("--log_dir", type=str, default='log_dir', help="directory to save logs")

    # Arguments for model configuration
    parser.add_argument("--model_name", type=str, default="./model/finbert", help="pretrained model used")
    parser.add_argument("--version", type=int, default=0, help="version number for saving or restoring models")

    # Arguments for data configuration
    parser.add_argument("--test_fold", type=int, default=0, help="which fold to use as test set")

    # Arguments for classic training configuration
    parser.add_argument("--num_epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--step_size", type=int, default=20000, help="step size for scheduler")
    parser.add_argument("--reasoner_config_num_labels", type=int, default=3, help="number of labels for the reasoner model")
    parser.add_argument("--gpus", type=int, nargs="+", required=True, help="list of GPU ids to use")
    JudgeModule.add_specific_args(parser)
    ReasonerModule.add_specific_args(parser)

    # Arguments for training configuration specific to CogLTX
    parser.add_argument("--num_samples", type=str, default="1,1,1,1", help="number of continuous, discrete random samples and promising samples for judge training")
    parser.add_argument("--latent", action="store_true", help="true if no relevance labels provided")
    parser.add_argument("--init_relevance", type=str, default="glove", help="initial relevance method, e.g., bm25 or glove")

    # Arguments for inference configuration
    parser.add_argument("--only_predict", action="store_true", help="true if only doing prediction")
    parser.add_argument("--times", type=str, default="3,5", help="MemRecall times")
    parser.add_argument("--batch_size_inference", type=int, default=8, help="batch size used in MemRecall")

    return parser

if __name__ == "__main__":
    # Set up argument parser and read
    parser = ArgumentParser(add_help=False)
    parser = set_parser_arguments(parser)
    config = parser.parse_args()

    # config.save_dir = str(output_dir / f"fold_{config.test_fold}" / config.save_dir)
    # config.tmp_dir = str(output_dir / f"fold_{config.test_fold}" / config.tmp_dir)
    # config.log_dir = str(output_dir / f"fold_{config.test_fold}" / config.log_dir)

    config.save_dir = str(output_dir / "final" / config.save_dir)
    config.tmp_dir = str(output_dir / "final" / config.tmp_dir)
    config.log_dir = str(output_dir / "final" / config.log_dir)

    # Get dataset needed 
    # train_dataset_filename = str(data_dir / f"mda_train_fold_{config.test_fold}.pkl")
    # val_dataset_filename = str(data_dir / f"mda_val_fold_{config.test_fold}.pkl")

    train_dataset_filename = str(data_dir / f"mda_train_final.pkl")
    test_dataset_filename = str(data_dir / f"mda_test_final.pkl")

    # Show fold processed
    #print(f"Processing fold {config.test_fold}...")

    # Train model 
    if not config.only_predict:
        train_model(config, train_dataset_filename)

    # Inference
    #result_dir = result_dir / f"fold_{config.test_fold}"
    #result_dir.mkdir(parents=True, exist_ok=True)

    label_names = ['negative', 'neutral', 'positive']
    results = []
    y_pred, y_true = [], []

    for qbuf, dbuf, buf, relevance_score, ids, output in prediction(config, test_dataset_filename):
        logits = output[0].view(-1)
        probs = F.softmax(logits, dim=0)

        gold = int(qbuf[0].label)
        pred = probs.argmax().item()

        # Next line for demo
        # print(f"\nPrediction for SHID_MDA_2021.txt: {label_names[pred]}")

        pred_neg, pred_neu, pred_pos = probs.tolist()

        y_pred.append(pred)
        y_true.append(gold)

        # Append to result df
        ticker = qbuf[0].filename.split('_MDA_')[0]
        year = int(qbuf[0].filename.split('_MDA_')[1].split('.txt')[0])

        # results.append({
        #     "Ticker": ticker,
        #     "Ticker_Fold": config.test_fold,
        #     "Year": year,
        #     "YoY_Close_Category": label_names[gold],
        #     "Tone_Label_Prob": label_names[pred],
        #     "Negative_Prob": pred_neg,
        #     "Neutral_Prob": pred_neu, 
        #     "Positive_Prob": pred_pos,
        # })

        results.append({
            "Ticker": ticker,
            "Year": year,
            "YoY_Close_Category": label_names[gold],
            "Tone_Label_Prob": label_names[pred],
            "Negative_Prob": pred_neg,
            "Neutral_Prob": pred_neu, 
            "Positive_Prob": pred_pos,
        })

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=[0,1,2], zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0)
    
    # Write results to file
    df = pd.DataFrame(results)
    df = df.sort_values(by=["Ticker", "Year"])
    # df.to_csv(result_dir / f"Fold_{config.test_fold}_Result_with_balanced_dataset_epoch_6.csv", index=False)
    df.to_csv(result_dir / f"Final_Result.csv", index=False)
    
    metrics = {
        "accuracy": accuracy,
        "macro": {
            "precision": precision_macro,
            "recall": recall_macro,
            "f1_score": f1_macro
        },
        "per_class": {
            "negative": {
                "precision": precision[0],
                "recall": recall[0],
                "f1_score": f1[0],
                "support": float(support[0])
            },
            "neutral": {
                "precision": precision[1],
                "recall": recall[1],
                "f1_score": f1[1],
                "support": float(support[1])
            },
            "positive": {
                "precision": precision[2],
                "recall": recall[2],
                "f1_score": f1[2],
                "support": float(support[2])
            }
        }
    } 

    # with open(result_dir / f"Fold_{config.test_fold}_Metrics_with_balanced_dataset_epoch_6.json", "w") as f:
    #     json.dump(metrics, f, indent=4)
    with open(result_dir / f"Final_Metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
         