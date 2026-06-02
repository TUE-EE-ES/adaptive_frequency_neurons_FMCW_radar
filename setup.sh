#!/bin/bash

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cd src/py_network
python3 setup.py build_ext --inplace