#!/bin/bash

source constants.sh

pip install -e .
pip uninstall -y ninja && pip install ninja
cd /home/wth/My_codes/doremi/flash-attention && python setup.py install
cd /home/wth/My_codes/doremi/flash-attention/csrc/fused_dense_lib && pip install .
cd /home/wth/My_codes/doremi/flash-attention/csrc/xentropy && pip install .
cd /home/wth/My_codes/doremi/flash-attention/csrc/rotary && pip install .
cd /home/wth/My_codes/doremi/flash-attention/csrc/layer_norm && pip install .


