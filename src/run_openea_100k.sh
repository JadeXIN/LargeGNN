#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=2 python main_ea.py config_zq/config_openea_100k.json EN_FR_100K_V1 721_5fold/1/ > log/fr100k

CUDA_VISIBLE_DEVICES=3 python main_ea.py config/config_openea_100k.json EN_FR_100K_V1 721_5fold/1/ > log/fr100k_15_base

#CUDA_VISIBLE_DEVICES=2 python main_ea.py config/config_100k.json EN_DE_100K_V1 721_5fold/1/ > log/de100k_10_base