import os
import os.path as osp
import argparse
from loguru import logger
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


def read_yaml_cfg_impl(config_file: str) -> DictConfig:
    """
    Parse yaml config file, and merge sub-yaml file within the `include` filed.
    """
    conf = OmegaConf.load(config_file)
    if conf.get("include") is not None:
        if isinstance(conf.include, str):
            include_files = [conf.include]
        elif isinstance(conf.include, ListConfig):
            include_files = conf.include
        else:
            raise TypeError(
                "Only str/ListConfig are allowed for `include` field.")
        for include_file in include_files:
            if include_file.startswith("."):
                # Interpolate the full path.
                cur_config_path = osp.dirname(osp.abspath(config_file))
                include_file_full_path = f"{cur_config_path}/{include_file}"
                sub_conf = read_yaml_cfg_impl(include_file_full_path)
            else:
                sub_conf = read_yaml_cfg_impl(include_file)
            conf = OmegaConf.merge(sub_conf, conf)
        conf.pop("include", None)
    return conf


def read_yaml_cfg(config_file: str, verbose=True) -> DictConfig:
    cfg = read_yaml_cfg_impl(config_file)
    if verbose:
        logger.debug(f"Config: \n{OmegaConf.to_yaml(cfg)}")
    return cfg


def paser_yaml_cfg():
    parser = argparse.ArgumentParser(prog='llm')
    parser.add_argument('-c',
                        '--config_file',
                        type=str,
                        help="The config file path")
    args = parser.parse_args()
    assert os.path.exists(
        args.config_file), f"yaml config file not exist {args.config_file}"
    return read_yaml_cfg(args.config_file)
