add-apt-repository ppa:deadsnakes/ppa
apt-get update
#apt upgrade -y
apt-get install python3.12
#apt install python3.12-dev
#apt install python3.12-venv
apt install python3.12-full
apt install python3-pip
python3.12 -m ensurepip --upgrade
#python3 -m pip install --upgrade pip
python3.12 -m pip install --upgrade pip
#pip install setuptools
#pip install --upgrade setuptools
python3.12 -m pip install -r requirements.txt

apt install wget
wget https://downloads.skewed.de/skewed-keyring/skewed-keyring_1.1_all_$(lsb_release -s -c).deb
dpkg -i skewed-keyring_1.1_all_$(lsb_release -s -c).deb
echo "deb [signed-by=/usr/share/keyrings/skewed-keyring.gpg] https://downloads.skewed.de/apt $(lsb_release -s -c) main" > /etc/apt/sources.list.d/skewed.list
apt-get update
apt-get install python3-graph-tool