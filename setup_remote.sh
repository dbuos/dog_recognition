source activate pytorch
python --version
pip install -r requirements.txt
sudo snap install nvtop
pip install -e .
cp jupyter_notebook_config.py /home/ubuntu/.jupyter/jupyter_notebook_config.py
