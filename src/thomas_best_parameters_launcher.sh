python src/remi_xgboost_2.py --xtrain data/processed/X_train_4.csv --ytrain data/processed/Y_train.csv --xval data/processed/X_train_4.csv --yval data/processed/Y_train.csv --dirpred data/prediction/hand_featured/ --dirmodel models/hand_featured/ --dirlog logs/hand_featured --seed 1 --nepoch 200 --eta 0.1 --max_depth 6 --subsample 0.8 --colsample_bytree 0.5 --scale_pos_weight 1 --max_delta_step 13.0


