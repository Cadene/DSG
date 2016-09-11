echo usual
python src/thomas_main.py -v 0.8 -x data/interim/X_train_features_1.csv -e data/interim/X_test_features_1.csv -y data/raw/Y_train.csv -a 0.01
#echo cross_val
#python src/thomas_main.py --cv 10 -x data/interim/X_train_features_1.csv -e data/interim/X_test_features_1.csv -y data/raw/Y_train.csv -a 0.01
