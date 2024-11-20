# pylint: disable=invalid-name, too-many-branches, line-too-long
"""Models for experiments and functions for setting them up"""

from importlib import import_module
import os

from omegaconf import OmegaConf, DictConfig, open_dict


def model_config_factory(cfg: DictConfig, optuna_trial=None):
    """Model config factory"""
    if cfg.mode.name == "tune":
        model_config = get_tune_config(cfg, optuna_trial=optuna_trial)
    elif cfg.mode.name == "exp":
        model_config = get_best_config(cfg)
    else:
        raise NotImplementedError

    return model_config


def get_tune_config(cfg: DictConfig, optuna_trial=None):
    """Returns random HPs defined by the models random_HPs() function"""
    if "tunable" in cfg.model:
        assert cfg.model.tunable, "Model is specified as not tunable, aborting"

    try:
        model_module = import_module(f"src.models.{cfg.model.name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"No module named '{cfg.model.name}' \
                                  found in 'src.models'. Check if model name \
                                  in config file and its module name are the same"
        ) from e

    try:
        random_HPs = model_module.random_HPs
    except AttributeError as e:
        raise AttributeError(
            f"'src.models.{cfg.model.name}' has no function\
                             'random_HPs'. Is the model not supposed to be\
                             tuned, or the function misnamed/not defined?"
        ) from e

    model_cfg = random_HPs(cfg, optuna_trial=optuna_trial)

    print("Tuning model config:")
    print(f"{OmegaConf.to_yaml(model_cfg)}")

    return model_cfg


def get_best_config(cfg: DictConfig):
    """
    1. If cfg.HP_path is not None, return the HPs stored in cfg.HP_path.
    2. Else return the HPs defined by 'default_HPs(cfg)' function in the model's .py module
    """
    if "HP_path" in cfg and cfg.HP_path is not None:
        # 1. try to load config from cfg.HP_path.
        # Note: dataset shape information is loaded as 'data_info' key,
        # just in case
        assert cfg.HP_path.endswith(".json") or cfg.HP_path.endswith(".yaml"), f"'{cfg.HP_path}' \
            is not json or yaml file, aborting"
        try:
            with open(cfg.HP_path, "r", encoding="utf8") as f:
                model_cfg = OmegaConf.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"'{cfg.HP_path}' config is not found"
            ) from e

        with open_dict(model_cfg):
            model_cfg.data_info = cfg.dataset.data_info

    else:
        # 2. try to get the HPs defined by 'default_HPs(cfg)' function in the model's .py module
        try:
            model_module = import_module(f"src.models.{cfg.model.name}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"No module named '{cfg.model.name}' \
                                    found in 'src.models'. Check if model name \
                                    in config file and its module name are the same"
            ) from e

        try:
            default_HPs = model_module.default_HPs
        except AttributeError as e:
            raise AttributeError(
                f"'src.models.{cfg.model.name}' has no function\
                                'default_HPs'. Is the function misnamed/not defined?"
            ) from e

        model_cfg = default_HPs(cfg)

    print("Loaded model config:")
    print(f"{OmegaConf.to_yaml(model_cfg)}")
    return model_cfg


def model_factory(cfg: DictConfig, model_cfg: DictConfig):
    """Models factory"""
    try:
        model_module = import_module(f"src.models.{cfg.model.name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"No module named '{cfg.model.name}' \
                                  found in 'src.models'. Check if model name \
                                  in config file and its module name are the same"
        ) from e

    try:
        get_model = model_module.get_model
    except AttributeError as e:
        raise AttributeError(
            f"'src.models.{cfg.model.name}' has no function\
                             'get_model'. Is the function misnamed/not defined?"
        ) from e

    model = get_model(cfg, model_cfg)

    return model
