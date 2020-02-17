# bash hw3_test.sh <testing data folder> <prediction.csv>
rm hw3_model.pth
wget https://www.dropbox.com/s/ghoielqsibccy5s/hw3_model.pth
python3 hw3_test.py -i $1 -o $2 -m hw3_model.pth
