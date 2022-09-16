#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 python main_baseline.py config/config_1m.json EN_FR_1M "" >log/fr1m_30_largeea