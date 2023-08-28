import socketserver
import sys

from argparse import ArgumentParser
from omegaconf import OmegaConf
from torch.cuda import device_count
from torch.multiprocessing import spawn
from typing import List

from src.config_torchwavenet import Config, parse_configs
from src.learner_torchwavenet import train, train_distributed


def _get_free_port():
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]

all_datasets = ['QPSK_CommSignal2', 'QPSK2_CommSignal2', 'QAM16_CommSignal2', 'OFDMQPSK_CommSignal2',
                'QPSK_CommSignal3', 'QPSK2_CommSignal3', 'QAM16_CommSignal3', 'OFDMQPSK_CommSignal3', 'CommSignal2_CommSignal3',
                'QPSK_EMISignal1', 'QPSK2_EMISignal1', 'QAM16_EMISignal1', 'OFDMQPSK_EMISignal1', 'CommSignal2_EMISignal1',
                'QPSK_CommSignal5G1', 'QPSK2_CommSignal5G1', 'QAM16_CommSignal5G1', 'OFDMQPSK_CommSignal5G1', 'CommSignal2_CommSignal5G1']

def main(argv: List[str]):
    parser = ArgumentParser(description="Train a Diffwave model.")
    parser.add_argument("--sigindex", type=int, required=True,
                        help="Index for Mixture Type.")
    parser.add_argument("--config", type=str, default="src/configs/wavenet.yml",
                        help="Configuration file for model.")
    args = parser.parse_args(argv[1:-1])
    
    sigtype = all_datasets[args.sigindex]
    # First create the base config
    cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_cli(
        argv[-1].split("::")) if argv[-1] != "" else None
    cfg: Config = Config(**parse_configs(cfg, cli_cfg))
    cfg.data.root_dir = f"npydataset/Dataset_{sigtype}_Mixture"
    cfg.model_dir = f"torchmodels/dataset_{sigtype.lower()}_mixture_wavenet"
    
    # Setup training
    world_size = device_count()
    if world_size != cfg.distributed.world_size:
        raise ValueError(
            "Requested world size is not the same as number of visible GPUs.")
    if cfg.distributed.distributed:
        if world_size < 2:
            raise ValueError(
                "Distributed training cannot be run on machine"
                f" with {world_size} device(s)."
            )
        if cfg.data.batch_size % world_size != 0:
            raise ValueError(
                f"Batch size {cfg.data.batch_size} is not evenly"
                f" divisble by # GPUs = {world_size}."
            )
        cfg.data.batch_size = cfg.data.batch_size // world_size
        port = _get_free_port()
        spawn(
            train_distributed, 
            args=(world_size, port, cfg), 
            nprocs=world_size, 
            join=True
        )
    else:
        train(cfg)


if __name__ == "__main__":
    argv = sys.argv
    if len(sys.argv) == 3:
        argv = argv + [""]
    main(argv)
