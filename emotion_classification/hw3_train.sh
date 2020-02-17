# bash hw3_train.sh <training data folder>  <train.csv>
python3 hw3_train.py -i $1 -l $2 -o models/hw3_model.pth -e 200
