# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

python -m spacy download en_core_web_sm
wget -r --no-parent https://github.com/TonyNemo/UBAR-MultiWOZ/archive/refs/heads/master.zip
mv ./github.com/TonyNemo/UBAR-MultiWOZ/archive/refs/heads/master.zip .
rm -r github.com
unzip master.zip
rm master.zip
cd UBAR-MultiWOZ-master
python data_analysis.py
python preprocess.py 
cd .. 
mv ./UBAR-MultiWOZ-master/data .
mv ./UBAR-MultiWOZ-master/db ./data/
rm -r UBAR-MultiWOZ-master
mv data/ ./ubar-preprocessing/
cp -r ./ubar-preprocessing/data/db ./ubar-preprocessing