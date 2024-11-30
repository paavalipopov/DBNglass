# pylint: disable=too-many-statements, too-many-locals, invalid-name, unbalanced-tuple-unpacking, no-value-for-parameter
"""Script for running experiments: tuning and testing hypertuned models"""
import os
from copy import deepcopy

from omegaconf import OmegaConf, DictConfig
import hydra

import pandas as pd
import numpy as np

import optuna

from src.utils import set_project_name, set_run_name, validate_config, get_resume_params
from src.data import data_factory, data_postfactory
from src.dataloader import dataloader_factory, cross_validation_split
from src.model import model_config_factory, model_factory
from src.model_utils import optimizer_factory, scheduler_factory
from src.trainer import trainer_factory


@hydra.main(version_base=None, config_path="../src/conf", config_name="exp_config")
def start(cfg: DictConfig):
    """Main script for starting experiments"""

    # check if config is correct
    validate_config(cfg)

    # set project name and directory
    set_project_name(cfg)
    # load interrupted config
    if "resume" in cfg and cfg.resume:
        cfg = get_resume_params(cfg)
    else:
        os.makedirs(cfg.project_dir, exist_ok=True)
    with open(f"{cfg.project_dir}/general_config.yaml", "w", encoding="utf8") as f:
        OmegaConf.save(cfg, f)

    print("General config:")
    print(OmegaConf.to_yaml(cfg))

    # load dataset, compute FNCs if model requires them.
    original_data = data_factory(cfg)

    if cfg.mode.name == "tune":
        if ("single_HPs" in cfg and cfg.single_HPs) or (
            "tuning_holdout" in cfg.dataset and cfg.dataset.tuning_holdout
        ):
            # use the whole original_data for tuning.
            # the obtained HPs are better to be used on other datasets
            # (unless cfg.dataset.tuning_holdout is True),
            # otherwise there is a testing-on-train-data danger

            # resume flags check
            is_interupted = "resume" in cfg and cfg.resume

            tune(cfg=cfg, original_data=original_data, is_interupted=is_interupted)

        else:
            # tune each original_data CV fold independently

            # set starting k
            if "resume" in cfg and cfg.resume:
                starting_k = cfg.resumed_params.outer_k
            else:
                starting_k = 0

            for k in range(starting_k, cfg.mode.n_splits):
                # resume flags check
                is_interupted = "resume" in cfg and cfg.resume and k == starting_k

                tune_fold_data = deepcopy(original_data)
                tune_fold_data["main"], _ = cross_validation_split(
                    tune_fold_data["main"], cfg.mode.n_splits, k
                )
                tune(
                    cfg=cfg,
                    original_data=tune_fold_data,
                    outer_k=k,
                    is_interupted=is_interupted,
                )

    elif cfg.mode.name == "exp":
        # resume flags check
        is_interupted = "resume" in cfg and cfg.resume

        experiment(cfg=cfg, original_data=original_data, is_interupted=is_interupted)


def tune(cfg, original_data, outer_k=None, is_interupted=False):
    """Given config and data, run several cross-validated rounds of optimal HP search"""

    # set starting trial
    if is_interupted:
        starting_trial = cfg.resumed_params.trial
    else:
        starting_trial = 0

    tuner = OptunaTuner(cfg, original_data, outer_k)
    tuner.optimize()
    # # for each trial get new set of HPs, test them using CV

    # get optimal config and save it
    df = pd.read_csv(f"{cfg.k_dir}/trial_runs.csv")
    best_idx = df["score"].idxmax()
    best_config_path = df.loc[best_idx]["path_to_config"]
    with open(best_config_path, "r", encoding="utf8") as f:
        best_config = OmegaConf.load(f)
    with open(f"{cfg.k_dir}/best_config.yaml", "w", encoding="utf8") as f:
        OmegaConf.save(best_config, f)

class OptunaTuner:
    def __init__(self, cfg, data, outer_k=None) -> None:
        self.cfg = cfg
        self.data = data
        self.outer_k = outer_k
    
    def optimize(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.cfg.mode.n_trials)

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    def objective(self, trial):
        # get random model config
        model_cfg = model_config_factory(self.cfg, optuna_trial=trial)
        # reshape data according to model config (if needed)
        data = data_postfactory(
            self.cfg,
            model_cfg,
            self.data,
        )
        # run nested CV
        for inner_k in range(0, self.cfg.mode.n_splits):
            if self.outer_k is not None:
                print(f"Outer k: {self.outer_k:02d}")
            print(f"Trial: {trial.number:04d}")
            print(f"Inner k: {inner_k:02d}")

            set_run_name(self.cfg, outer_k=self.outer_k, trial=trial.number, inner_k=inner_k)
            os.makedirs(self.cfg.run_dir, exist_ok=True)
            dataloaders = dataloader_factory(self.cfg, data, k=inner_k)
            results = run_trial(self.cfg, model_cfg, dataloaders)

            # save results of nested CV in the trial directory
            df = pd.DataFrame(results, index=[0])
            with open(f"{self.cfg.trial_dir}/CV_runs.csv", "a", encoding="utf8") as f:
                df.to_csv(f, header=f.tell() == 0, index=False)

        # save model config
        with open(f"{self.cfg.trial_dir}/model_config.yaml", "w", encoding="utf8") as f:
            OmegaConf.save(model_cfg, f)

        # summarize the trial's CV results and save them
        df = pd.read_csv(f"{self.cfg.trial_dir}/CV_runs.csv")
        score = np.mean(df["test_score"].to_numpy())
        loss = np.mean(df["test_average_loss"].to_numpy())
        time = np.mean(df["training_time"].to_numpy())
        df = pd.DataFrame(
            {
                "trial": trial.number,
                "score": score,
                "loss": loss,
                "time": time,
                "path_to_config": f"{self.cfg.trial_dir}/model_config.yaml",
            },
            index=[0],
        )
        with open(f"{self.cfg.k_dir}/trial_runs.csv", "a", encoding="utf8") as f:
            df.to_csv(f, header=f.tell() == 0, index=False)
        
        return score


def experiment(cfg, original_data, is_interupted=False):
    """Given config and data, run cross-validated rounds with optimal HPs"""

    # set starting k
    if is_interupted:
        starting_k = cfg.resumed_params.outer_k
    else:
        starting_k = 0

    for outer_k in range(starting_k, cfg.mode.n_splits):
        # for each fold get optimal set of HPs,
        # unless single_HP is True,
        # or model HPs are hardcoded

        # set starting trial
        if is_interupted and outer_k == starting_k:
            starting_trial = cfg.resumed_params.trial
        else:
            starting_trial = 0

        model_cfg = model_config_factory(cfg, outer_k)
        # reshape data according to model config (if needed)
        data = data_postfactory(
            cfg,
            model_cfg,
            original_data,
        )
        # for outer_k test fold, train model n_trials times,
        # using different train/valid split each time
        for trial in range(starting_trial, cfg.mode.n_trials):
            print(f"k: {outer_k:02d}")
            print(f"Trial: {trial:04d}")

            set_run_name(cfg, outer_k=outer_k, trial=trial)
            os.makedirs(cfg.run_dir, exist_ok=True)
            dataloaders = dataloader_factory(cfg, data, k=outer_k, trial=trial)
            results = run_trial(cfg, model_cfg, dataloaders)

            # save run's results in the folds directory
            df = pd.DataFrame(results, index=[0])
            with open(f"{cfg.k_dir}/fold_runs.csv", "a", encoding="utf8") as f:
                df.to_csv(f, header=f.tell() == 0, index=False)

        # save outer_k's model config
        with open(f"{cfg.k_dir}/model_config.yaml", "w", encoding="utf8") as f:
            OmegaConf.save(model_cfg, f)

        # load and save the fold's results in the project directory
        df = pd.read_csv(f"{cfg.k_dir}/fold_runs.csv")
        with open(f"{cfg.project_dir}/runs.csv", "a", encoding="utf8") as f:
            df.to_csv(f, header=f.tell() == 0, index=False)


def run_trial(cfg, model_cfg, dataloaders):
    """Given config and prepared dataloaders, build and train the model and return test results"""
    model = model_factory(cfg, model_cfg)
    optimizer = optimizer_factory(cfg, model_cfg, model)
    scheduler = scheduler_factory(cfg, model_cfg, optimizer)

    trainer = trainer_factory(
        cfg,
        model_cfg,
        dataloaders,
        model,
        optimizer,
        scheduler,
    )

    results = trainer.run()

    return results


if __name__ == "__main__":
    start()
