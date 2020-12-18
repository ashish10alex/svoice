#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Yossi Adi (adiyoss)

tr_path=egs/librimix_dataset/tr
val_path=egs/librimix_dataset/vl
test_path=egs/librimix_dataset/tt



if [[ ! -e $tr_path ]]; then
    mkdir -p $tr_path
fi

if [[ ! -e $val_path ]]; then
    mkdir -p $val_path
fi

if [[ ! -e $test_path ]]; then
    mkdir -p $test_path
fi



#python -m svoice.data.audio /jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/asteroid/egs/librimix/ConvTasNet/data/Libri2Mix/wav8k/min/train-100/mix_both > $tr_path/mix.json
#python -m svoice.data.audio /jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/asteroid/egs/librimix/ConvTasNet/data/Libri2Mix/wav8k/min/train-100/s1 > $tr_path/s1.json
#python -m svoice.data.audio /jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/asteroid/egs/librimix/ConvTasNet/data/Libri2Mix/wav8k/min/train-100/s2 > $tr_path/s2.json


python -m svoice.data.audio /jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/asteroid/egs/librimix/ConvTasNet/data/Libri2Mix/wav8k/min/dev/mix_both > $val_path/mix.json
python -m svoice.data.audio /jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/asteroid/egs/librimix/ConvTasNet/data/Libri2Mix/wav8k/min/dev/s1 > $val_path/s1.json
python -m svoice.data.audio /jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/asteroid/egs/librimix/ConvTasNet/data/Libri2Mix/wav8k/min/dev/s2 > $val_path/s2.json

python -m svoice.data.audio /jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/asteroid/egs/librimix/ConvTasNet/data/Libri2Mix/wav8k/min/test/mix_both > $test_path/mix.json
python -m svoice.data.audio /jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/asteroid/egs/librimix/ConvTasNet/data/Libri2Mix/wav8k/min/test/s1 > $test_path/s1.json
python -m svoice.data.audio /jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/asteroid/egs/librimix/ConvTasNet/data/Libri2Mix/wav8k/min/test/s2 > $test_path/s2.json
