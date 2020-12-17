#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Yossi Adi (adiyoss)

path=egs/librimix_dataset/tr
if [[ ! -e $path ]]; then
    mkdir -p $path
fi
python -m svoice.data.audio /jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/asteroid/egs/librimix/ConvTasNet/data/Libri2Mix/wav8k/min/train-100/mix_both > $path/mix.json
python -m svoice.data.audio /jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/asteroid/egs/librimix/ConvTasNet/data/Libri2Mix/wav8k/min/train-100/s1 > $path/s1.json
python -m svoice.data.audio /jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/asteroid/egs/librimix/ConvTasNet/data/Libri2Mix/wav8k/min/train-100/s2 > $path/s2.json
