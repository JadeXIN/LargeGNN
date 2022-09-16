#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python main_baseline.py config/config_openea_15k.json EN_FR_15K_V1 721_5fold/1/ >log/fr15k_40_largeea
#CUDA_VISIBLE_DEVICES=2 python main_ea.py config/config_15k.json EN_DE_15K_V1 721_5fold/1/ >log/fr15k_40_base
