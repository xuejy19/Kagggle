'''
Description: titanic_survival_predict
Version: V1.0
Author: xuejy19@mails.tsinghua.edu.cn
Date: 2020-08-30 09:41:49
'''
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import os 

def read_data(dataset_path):
    for root,dirs,files in os.walk(DATA_PATH):
        for file in files:
            if file == 'train.csv':
                train_data = pd.read_csv(os.path.join(DATA_PATH,file)) 
            elif file == 'test.csv':
                test_data = pd.read_csv(os.path.join(DATA_PATH,file)) 
    return train_data, test_data

if __name__ == '__main__':
    # for root,_,filename in os.walk()
    ROOT_PATH = os.path.join(os.getcwd(),'..')
    DATA_PATH = os.path.join(ROOT_PATH,'datasets')
    train_data, test_data = read_data(DATA_PATH)
    # train_women = train_data.loc[train_data.Sex == 'female']["Survived"]
    # train_men = train_data.loc[train_data.Sex == 'male']["Survived"]
    # rate_women = sum(train_women)/len(train_women)
    # rate_men = sum(train_men)/len(train_men)
    # print("女性存活率 = " + np.str(rate_women))
    # print("男性存活率 = " + np.str(rate_men))
    train_label = train_data["Survived"]
    Select_features = ['Pclass', 'Sex', 'SibSp', 'Parch']
    train_feature = pd.get_dummies(train_data[Select_features])
    test_feature = pd.get_dummies(test_data[Select_features])
    model = RandomForestClassifier(n_estimators=1000, random_state= 20)
    model.fit(train_feature,train_label)
    predictions = model.predict(test_feature)
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('my_submission.csv', index=False)
    print("Your submission was successfully saved!")
