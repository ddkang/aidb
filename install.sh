# setup miniconda
mkdir -p ./miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda3/miniconda.sh
bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3
rm -rf ./miniconda3/miniconda.sh
export PATH=$./miniconda3/bin:$PATH
./miniconda3/bin/conda init bash
. ~/.bashrc

# setup python environment
conda create -y -n aidb python=3.9
conda activate aidb
conda install -y python=3.9
pip install -r requirements.txt

mkdir -p ./tests/vldb_tests/data
git clone https://github.com/ttt-77/AIDB_data.git ./tests/vldb_tests/data
gdown 'https://drive.google.com/uc?id=1IobfQq2AtpaB74PhW5nEHaZj_G8VIhPE'
unzip embedding.zip -d ./tests/vldb_tests/data