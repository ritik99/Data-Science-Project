import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from collections import namedtuple
from sklearn.preprocessing import StandardScaler


def load_adult(smaller=False, scaler=True):
    '''
    :param smaller: selecting this flag it is possible to generate a smaller version of the training and test sets.
    :param scaler: if True it applies a StandardScaler() (from sklearn.preprocessing) to the data.
    :return: train and test data.

    Features of the Adult dataset:
    0. age: continuous.
    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. fnlwgt: continuous.
    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
    Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. education-num: continuous.
    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
    Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
    Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.
    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.
    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    (14. label: <=50K, >50K)
    '''
    data = pd.read_csv(
        "7_again_gender_hashed_train_data.csv",
        #names=[
        #    "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        #    "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
        #    "hours per week", "native-country", "income"]
            )
    data_test = pd.read_csv(
        "7_again_gender_hashed_test_data.csv",
        #names=[
        #    "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        #    "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
        #    "hours per week", "native-country", "income"]
    )
    len_train = data.shape[0]
    #print (len_train)
    data = pd.concat([data, data_test])
    # Considering the relative low portion of missing data, we discard rows with missing data
    
    datamat = data.values
    
    #target = np.array([-1.0 if val == 0 else 1.0 for val in np.array(datamat)[:, -1]])
    target = data["income"]
    
    datamat = datamat[:, :-1]
    #print (datamat.shape)
    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)
    if smaller:
        print('A smaller version of the dataset is loaded...')
        data = namedtuple('_', 'data, target')(datamat[:len_train // 20, :], target[:len_train // 20])
        data_test = namedtuple('_', 'data, target')(datamat[len_train:, :], target[len_train:])
    else:
        print('The dataset is loaded...')
        data = namedtuple('_', 'data, target')(datamat[:len_train, :], target[:len_train])
        data_test = namedtuple('_', 'data, target')(datamat[len_train:, :], target[len_train:])
    return data, data_test
