# setup miniconda
mkdir -p /projects/illinois/eng/cs/ddkang/tengjun2/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /projects/illinois/eng/cs/ddkang/tengjun2/miniconda3/miniconda.sh
bash /projects/illinois/eng/cs/ddkang/tengjun2/miniconda3/miniconda.sh -b -u -p /projects/illinois/eng/cs/ddkang/tengjun2/miniconda3
rm -rf /projects/illinois/eng/cs/ddkang/tengjun2/miniconda3/miniconda.sh
export PATH=$/projects/illinois/eng/cs/ddkang/tengjun2/miniconda3/bin:$PATH
/projects/illinois/eng/cs/ddkang/tengjun2/miniconda3/bin/conda init bash
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