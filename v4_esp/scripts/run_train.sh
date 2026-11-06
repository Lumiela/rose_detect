#!/bin/bash
cd /helpai-storage/flower_git/v4_esp/scripts
/usr/bin/python 20_esp_export.py
for s in 0 1 2; do /usr/bin/python 30_esp_train.py ft_s$s $s; done
echo ALL_DONE
