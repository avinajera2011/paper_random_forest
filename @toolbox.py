import os, sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from pandas import Timestamp
from datetime import timedelta
from collections import Counter
from itertools import repeat
import joblib

models = {
        1: 'bag_clf',
        2: 'pas_clf',
        3: 'adaboost',
        4: 'bag_clf-pas_clf',
        5: 'bag_clf-adaboost',
        6: 'pas_clf-adaboost',
        7: 'bag_clf-past_clf-adaboost',      
        8: 'rf'
            }
k_data = {1: 'relabeled', 2: 'original'}   
months = ('January', 'February','March','April','May','June','July','August','September','October', 'November', 'December')
init_month = 1
data_x_mz = pd.read_excel("todo_ok.xlsx", sheet_name='Datos por MZ', usecols=['MZ', 'Norte', 'Sur', 'Oeste', 'Este'],
                          dtype={'MZ': str, 'Norte': str, 'Sur': str, 'Oeste': str, 'Este': str})
scaler = MinMaxScaler(feature_range=(0, 1))
result_to_txt=list()
width_of_text = 80

def select_model() -> str:
    """_summary_
    Get the model to be used
    Returns: Name of the model to be used
        _type_: _description_
    """
    model_id = ''
    while not model_id.isdigit() :
        model_id = input('The following models can be selected to train and test it:\n'
                        '\n1 Bagging Classifier with Random Forest (bag_clf)' 
                        '\n2 Pasting Classifier with Random Forest (pas_clf)' 
                        '\n3 AdaBoost Classifier with Random Forest (AdaBoost)'
                        '\n4 bag_clf - pas_clf'
                        '\n5 bag_clf - AdaBoost'
                        '\n6 pas_clf - AdaBoost'
                        '\n7 bag_clf - pas_clf - AdaBoost'
                        '\n\nPlease type the number of the model:')
        if model_id.isdigit():
            if int(model_id) > 7 or int(model_id) < 1:
                model_id = ''

    print('The selected model is:', models.get(int(model_id)))
    return models.get(int(model_id))

def select_data() -> str:
    """_summary_
    Get the kind of data to be used
    Returns: Kind the data to be used
        _type_: _description_
    """
    kind_data_set = ''
    while not kind_data_set.isdigit():
        kind_data_set = input('There are two scenarios for the data:\n\n'
                              '1 Data of blocks identified as negative will be relabeled\n'
                              '2 Original data (no data will be relabeled)\n'
                              '\n\nPlease type the number of the data to be used:')
        if kind_data_set.isdigit():
            if int(kind_data_set) < 1 or int(kind_data_set) > 2:
                kind_data_set = ''
    return k_data.get(int(kind_data_set))
        

def process_data(k_data: str) -> pd.DataFrame:
    """_summary_
    Process the data for the model
    Args:
        k_data (str): kind of data to be used
    Returns: DataFrame ready to be used
        pd.DataFrame: _description_
    """                             
    csv_data = pd.read_csv('ok_data.csv', parse_dates=[2], index_col=0)
    print("Data have been load")
    # Relabel data if necessary
    if k_data != 'relabeled':
        full_data = csv_data.copy()
    else:
        full_data = relabel_data(csv_data)
    return full_data


def relabel_data(pd_data: pd.DataFrame) -> pd.DataFrame:
    """_summary_
    Relabel all blocks identified as negative
    Args:
        pd_data (pd.DataFrame): DataFrame with all data
    Returns:
        pd.DataFrame: _description_
    """
    print('Relabeling all blocks identified as negative')
    init_date = pd.to_datetime('3/1/2019', format="%m/%d/%Y")
    full_data = pd_data.copy()
    full_data_cluster = full_data[full_data.date>=init_date]
    data_to_cluster = full_data_cluster[(full_data_cluster['status']==0)].loc[:, '1_week_ago':'8_week_ago']
    km=KMeans(n_clusters=2, random_state=0)
    km.fit(scaler.fit_transform(data_to_cluster), full_data_cluster.status)
    neg = full_data_cluster[full_data_cluster['status'] == 0]
    neg.insert(neg.shape[1], 'status_clf', km.labels_, True)
    pos = full_data_cluster[full_data_cluster['status'] == 1]
    pos.insert(pos.shape[1], 'status_clf', np.ones(pos.shape[0]).reshape((-1, 1)), True)
    full_data = pd.concat([neg, pos])
    print('All blocks identified as negative has been relabeled')
    return full_data
    
def get_estimators_table() -> tuple:
    """_summary_
    Get estimators' values
    Returns:
        tuple: tuple of values of each estimators
    """
    estimators_table = pd.read_csv("best_estimators.csv", index_col=0)
    rf_estimator = estimators_table.best_data_original['rf']
    bag_estimator = estimators_table.best_data_original['bag_clf']
    pas_estimator = estimators_table.best_data_original['bag_clf']
    ada_estimator = estimators_table.best_data_original['adaboost']
    return rf_estimator, bag_estimator, pas_estimator, ada_estimator, estimators_table


def train_model(model_name: str, full_data:pd.DataFrame, k_data: str) -> StackingClassifier:
    """_summary_
        Train selected model
    Args:
        model_name (str): name of model to train
        full_data (pd.DataFrame): Dataframe with all data
        k_data (str): kind of data for training (original or relabeled all block identified as negative)

    Returns:
        StackingClassifier: _description_
    """
    est_table = get_estimators_table()[-1]
    if k_data == 'relabeled':
        rf_estimator, bag_estimator, pas_estimator, ada_estimator = est_table.best_data_labeled
        rf_max_leaf, bag_max_leaf, pas_max_leaf, ada_max_leaf = est_table.max_leaf_data_labeled
    else:
        rf_estimator, bag_estimator, pas_estimator, ada_estimator = est_table.best_data_original
        rf_max_leaf, bag_max_leaf, pas_max_leaf, ada_max_leaf = est_table.max_leaf_data_original
    print_sms("The parameters for each model was taken from table.")
    raf_clf = RandomForestClassifier(max_leaf_nodes=rf_max_leaf, n_estimators=rf_estimator, random_state=42, n_jobs=-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    print_sms('Training model')
    if model_name == 'bag_clf-past_clf-adaboost':
        ada_clf = AdaBoostClassifier(raf_clf, n_estimators=ada_estimator, algorithm="SAMME.R", learning_rate=0.05)
        bag_clf = BaggingClassifier(RandomForestClassifier(), n_estimators=bag_estimator, bootstrap=True, n_jobs=-1)
        pas_clf = BaggingClassifier(RandomForestClassifier(), n_estimators=pas_estimator, bootstrap=False, n_jobs=-1)
        voting_clf = StackingClassifier([('bag', bag_clf), ('pas', pas_clf), ('ada', ada_clf)],
                                        final_estimator=RandomForestClassifier(random_state=43, n_jobs=-1))
    elif model_name == 'pas_clf-adaboost':
        ada_clf = AdaBoostClassifier(raf_clf, n_estimators=ada_estimator, algorithm="SAMME.R", learning_rate=0.05)
        pas_clf = BaggingClassifier(RandomForestClassifier(), n_estimators=pas_estimator, bootstrap=False, n_jobs=-1)
        voting_clf = StackingClassifier([('pas', pas_clf), ('ada', ada_clf)],
                                        final_estimator=RandomForestClassifier(random_state=43, n_jobs=-1))
    elif model_name == 'bag_clf-adaboost':
        ada_clf = AdaBoostClassifier(raf_clf, n_estimators=ada_estimator, algorithm="SAMME.R", learning_rate=0.05)
        bag_clf = BaggingClassifier(RandomForestClassifier(), n_estimators=bag_estimator, bootstrap=True, n_jobs=-1)
        voting_clf = StackingClassifier([('bag', bag_clf), ('ada', ada_clf)],
                                        final_estimator=RandomForestClassifier(random_state=43, n_jobs=-1))
    elif model_name == 'bag_clf-pas_clf':
        bag_clf = BaggingClassifier(RandomForestClassifier(), n_estimators=bag_estimator, bootstrap=True, n_jobs=-1)
        pas_clf = BaggingClassifier(RandomForestClassifier(), n_estimators=pas_estimator, bootstrap=False, n_jobs=-1)
        voting_clf = StackingClassifier([('bag', bag_clf), ('pas', pas_clf)],
                                        final_estimator=RandomForestClassifier(random_state=43, n_jobs=-1))
    elif model_name == 'adaboost':
        raf_clf = RandomForestClassifier(max_leaf_nodes=ada_max_leaf, n_estimators=rf_estimator, random_state=42, n_jobs=-1)
        voting_clf = AdaBoostClassifier(raf_clf, n_estimators=ada_estimator, algorithm="SAMME.R", learning_rate=0.05)
    elif model_name == 'pas_clf':
        raf_clf = RandomForestClassifier(max_leaf_nodes=pas_max_leaf, n_estimators=rf_estimator, random_state=42, n_jobs=-1)
        voting_clf = BaggingClassifier(RandomForestClassifier(), n_estimators=pas_estimator, bootstrap=False, n_jobs=-1)
    else: # bag_clf
        raf_clf = RandomForestClassifier(max_leaf_nodes=bag_max_leaf, n_estimators=rf_estimator, random_state=42, n_jobs=-1)
        voting_clf = BaggingClassifier(RandomForestClassifier(), n_estimators=bag_estimator, bootstrap=True, n_jobs=-1)
    print_sms(f"starting training of model '{model_name}' at ", Timestamp.now())
    first_date = pd.to_datetime('3/1/2019', format="%m/%d/%Y")
    predict_date_begin = pd.to_datetime('1/1/2022', format="%m/%d/%Y")
    training_data = full_data[(full_data['date'] > first_date) & (full_data['date'] < predict_date_begin)]
    percent_training_set = training_data.shape[0] / full_data.shape[0]
    print_sms(f'Percentage of training set :{int(percent_training_set * 100)}')
    print_sms(f'Percentage of testing set :{100 - int(percent_training_set * 100)}')
    X = training_data.loc[:, '1_week_ago':'8_week_ago']
    if k_data != 'relabeled':
        y = training_data['status']
    else:
        y = training_data['status_clf']
    data_input = scaler.fit_transform(X)
    voting_clf.fit(data_input, y)
    print_sms('finished training at', Timestamp.now())
    return voting_clf
    
def test_model(voting_clf: StackingClassifier, full_data: pd.DataFrame) -> tuple:
    """_summary_
    Test the model in the testing set
    Args:
        voting_clf (StackingClassifier): StackingClassifier object
        full_data (pd.DataFrame): Dataframe with all data

    Returns:
        tuple: _description_
    """
    print_sms('Testing model')
    scaler = MinMaxScaler(feature_range=(0, 1))
    real_negative = list()
    rare_negative = list()
    real_positive = list()
    rare_positive = list()
    all_prediction_2022 = dict()
    from datetime import timedelta
    months = ('January', 'February','March','April','May','June','July','August','September','October', 'November', 'December')
    init_month = 1
    for i in range(init_month, 13):
        print(f'Testing the model on {months[i-1]} 2022')
        tmp_predict_date_begin = pd.to_datetime(f'{i}/1/2022', format="%m/%d/%Y")
        next_month_day = tmp_predict_date_begin + timedelta(days=31)
        tmp_predict_date_end = next_month_day - timedelta(days=next_month_day.day - 1)
        X_predict_data = full_data[(full_data['date'] >= tmp_predict_date_begin) & (full_data['date'] < tmp_predict_date_end)]
        X_predict = X_predict_data.loc[:, '1_week_ago':'8_week_ago']
        input_data = scaler.fit_transform(X_predict)
        vt_clf_prediction = voting_clf.predict(input_data)
        X_predict_data.insert(X_predict_data.shape[1], 'prediction', vt_clf_prediction, True)
        all_prediction_2022[months[i-1]] = tuple(X_predict_data[X_predict_data['prediction'] == 1].MZ)
        cf_mt = confusion_matrix(X_predict_data.status, vt_clf_prediction)
        real_negative.append( cf_mt[0,0])
        rare_negative.append(cf_mt[0,1])
        real_positive.append(cf_mt[1,1])
        rare_positive.append(cf_mt[1,0])
        voting_clf.fit(scaler.fit_transform(X_predict), X_predict_data.status)
    all_results = pd.DataFrame({'False_neg': rare_negative,
                           'True_neg': real_negative,
                           'False_pos': rare_positive,
                           'True_pos': real_positive})
    all_results.index = months[init_month-1:]
    print_sms("Testing finished at", Timestamp.now())
    return all_results, all_prediction_2022

def get_besides_zones(zone: str) -> dict:
    """
    Get the zones beside any given zone
    Args:
        zone: string with the number of the zone
    
    Returns:
        Dict: with the number on zone by cardinal points
    """
    beside = data_x_mz[data_x_mz['MZ'] == str(zone)]
    results = dict()
    results['Norte'] = __separate_beside_zones(beside['Norte'].iloc[0])
    results['Sur'] = __separate_beside_zones(beside['Sur'].iloc[0])
    results['Este'] = __separate_beside_zones(beside['Este'].iloc[0])
    results['Oeste'] = __separate_beside_zones(beside['Oeste'].iloc[0])
    return results

def __separate_beside_zones(zone: str) -> tuple:
    """
    Split beside zone
    
    Args: 
        zone: any string containing the data
    
    Retruns:
        tuple: all number of the zones
    """
    zone_str = zone
    if type(zone) is not str:
        zone_str = str(zone)
    split = (zone_str,)
    if len(zone_str) > 1:
        if ',' in zone_str:
            split = zone_str.split(',')
    return tuple(split)

def check_arround_zone_is_positives_future(besid_zone: str, first_week_pos_zone: tuple, pos_zone_after_1st_week: tuple):
    if besid_zone not in first_week_pos_zone and besid_zone in pos_zone_after_1st_week:
        return besid_zone

def get_model_performance(all_prediction_2022: dict, full_data: pd.DataFrame, all_results: pd.DataFrame) -> tuple:
    """_summary_

    Args:
        all_prediction_2022 (dict): Prediction of every month
        full_data (pd.DataFrame): full dataset
        all_results (pd.DataFrame): results of the model's testing

    Returns:
        tuple: (result by month, performance by month)
    """
    all_pred_keys = tuple(all_prediction_2022.keys())
    total_zones = len(Counter(full_data.MZ).keys())
    len(all_prediction_2022.get(all_pred_keys[0]))
    pos_per_1st_week = list()
    total_prediction_month_match = []
    real_month = []
    accuracy_per_month = []
    total_pos_model_month = list()
    pos_zone_future = list()
    len_pos = 0
    pos_zon_not_inspected_on_time = list()
    mz_pos_por_no_tratar_mz_adyacentes_previamente = list()
    pos_model_neg_real = list()
    pos_model_neg_real_pos_for_not_treat_arround_zones_previously = list()
    months = ('January', 'February','March','April','May','June','July','August','September','October', 'November', 'December')
    for i in range(init_month,13):
        date_begin = pd.to_datetime(f'{i}/1/2022', format="%m/%d/%Y")
        next_month_day = date_begin + timedelta(days=31)
        date_end = next_month_day - timedelta(days=next_month_day.day - 1)
        # Select just the data positive
        positive_zone_in_period = full_data[(full_data.date >= date_begin) & (full_data.date < date_end) & (full_data.status == 1)]
        if positive_zone_in_period.shape[0] > 1:
            # total positive zones predicted by model
            prediction_of_the_month = all_prediction_2022.get(months[i-1])
            total_pos_model_month.append(len(prediction_of_the_month))
            zones_matches = set(prediction_of_the_month).intersection(set(positive_zone_in_period.MZ))
            total_prediction_month_match.append(len(zones_matches))
            real_month.append(positive_zone_in_period.shape[0])
            acc_month = len(zones_matches) / positive_zone_in_period.shape[0]
            accuracy_per_month.append(round(acc_month, 4))
            # days_to_inspect_predicted_zones (dipz)
            dipz = round(len(prediction_of_the_month)/9) # AVG: 9 blocks inspected per day
            pos_zones_first_week = set(positive_zone_in_period[(positive_zone_in_period.date>=date_begin) &
                                                            (positive_zone_in_period.date<=date_begin + timedelta(days=dipz))].MZ)
            # Zones not inspected in first week according prediction (znifw)
            zones_not_detected = len(zones_matches.difference(pos_zones_first_week))
            pos_zon_not_inspected_on_time.append(zones_not_detected)
            # Positives zones due to posivites zones predicted by model not treated on time (pz)
            dif_pred_1st_week = set(zones_matches).difference(pos_zones_first_week)       
            for zone in dif_pred_1st_week:
                if len(str(zone)) == 1:
                    bes_zones = tuple(get_besides_zones(f'0{zone}').values())
                else:
                    bes_zones = tuple(get_besides_zones(zone).values())
                ok_bes_zones = []
                tmp =[]
                for card_point in bes_zones:
                    if len(card_point) == 1:
                        if card_point[0] != '-':
                            tmp.append(card_point[0])
                    else:
                        for it in card_point:
                            if card_point[0] != '-':
                                tmp.append(it)
                ok_bes_zones.extend(tmp)
                all_pos_zones_first_week = tuple([pos_zones_first_week for _ in range(len(ok_bes_zones))])
                all_dif_pred_1st_week = tuple([dif_pred_1st_week for _ in range(len(ok_bes_zones))])
                zone_in_future = list(map(check_arround_zone_is_positives_future, ok_bes_zones, 
                                        all_pos_zones_first_week, all_dif_pred_1st_week))
                if len(zone_in_future) > 1:       
                    for k in zone_in_future:
                        if k not in (None, '-') or type(k) in (list, tuple, str):
                            if type(k) is str:
                                pos_zone_future.append(k)
                            else:
                                for q in k:
                                    if type(q) in (list, tuple):
                                        pos_zone_future.extend(q)
                                    elif type(q) is str:
                                        pos_zone_future.append(q)
                    if len(pos_zone_future) > 1:
                        pos_zone_future = list(Counter(pos_zone_future).keys()) 
                    if '-' in pos_zone_future:
                        pos_zone_future.remove('-')
                elif type(zone_in_future) is str:
                    pos_zone_future.append(zone_in_future)
            if len(pos_zone_future) > len_pos:
                dif = len(pos_zone_future) - len_pos
                mz_pos_por_no_tratar_mz_adyacentes_previamente.append(dif)
            else:
                mz_pos_por_no_tratar_mz_adyacentes_previamente.append(0)
            len_pos = len(pos_zone_future)    
            # Total of zones predicted as positives and was negative in real data (PvN)
            dif_pos_model_vs_neg_real = set(prediction_of_the_month).difference(set(positive_zone_in_period.MZ))
            pos_model_neg_real.append(len(dif_pos_model_vs_neg_real))
    all_res = pd.DataFrame({'Zones_pred_model': total_pos_model_month,
                            'zones_detected': real_month, 'Zones_pred_model_match': total_prediction_month_match,
                            'model_accuracy': np.array(accuracy_per_month), 
                            # zones inspected on time (zion)
                            'zion': np.array(total_prediction_month_match) - np.array(pos_zon_not_inspected_on_time), #'match_1s_week': posi_first_week_match
                            'znifw': pos_zon_not_inspected_on_time,
                            'pz': mz_pos_por_no_tratar_mz_adyacentes_previamente,
                            'PvN': pos_model_neg_real
                        })
    all_res.index = months[init_month-1:]
    print_sms(all_res)
    print_sms(all_res.sum())
    print_sms(all_res.mean())
    report = all_results.copy()
    months = ('January', 'February','March','April','May','June','July','August','September','October', 'November', 'December')
    report.index = months[init_month-1:]
    good_classification = report.True_neg + report.True_pos
    total_classification = report.sum(axis=1)
    percentage = good_classification / total_classification
    report['tot_pron_corr'] = good_classification
    report['total_pron'] = total_classification
    report['percent_pron_corr'] = percentage
    print_sms('ACCURACY OF THE MODEL:', np.average(report.percent_pron_corr))
    print_sms(report)
    return all_res, report
    

def run_tool_box() -> tuple:
    """_summary_
    Get the choice of models to be used
    Returns:
        int: Number of choice selected
    """
    print_sms('Please, select one of the following options:')
    print_sms('1 Find best estimators (by GridSearchCV)')
    print_sms('2 Train and test model')
    print_sms('3 Predict')
    action = ''

    while not action.isdigit():
        action = input('Type here the selected option:')
        if action.isdigit():
            if int(action) < 1 or int(action) > 3:
                action = ''
    selected_action = int(action)
    print(''.ljust(width_of_text, '-'))
    if selected_action == 1:
        print_sms('The following models can be used:')
        print_sms('1 Bagging Classifier with Random Forest (bag_clf)')
        print_sms('2 Pasting Classifier with Random Forest (pas_clf)')
        print_sms('3 AdaBoost Classifier with Random Forest (AdaBoost)')
        print_sms('4 Random Forest (rf)')
        print_sms('With two different dataset')
        print_sms('a blocks identified as negative will be relabeled')
        print_sms('b original (no relabeled)')
    elif selected_action == 2:
        print_sms('The following models can be used:')
        print_sms('1 Bagging Classifier with Random Forest (bag_clf)')
        print_sms('2 Pasting Classifier with Random Forest (pas_clf)')
        print_sms('3 AdaBoost Classifier with Random Forest (AdaBoost)')
        print_sms('4 bag_clf - pas_clf')
        print_sms('5 bag_clf - AdaBoost')
        print_sms('6 pas_clf - AdaBoost')
        print_sms('7 bag_clf - pas_clf - AdaBoost')
        print_sms('With two different dataset')
        print_sms('a blocks identified as negative will be relabeled')
        print_sms('b original (no relabeled)')
    models_to_train = ''    
    incorrect_models, correct_models = repeat(list(), 2)
    while len(incorrect_models) > 0 or len(correct_models) == 0:
        print_sms('Please type the combination of models and kind of data')
        print_sms('For only one model (e.g: 1a)')
        if selected_action == 1:
            print_sms('For more than one model (e.g: 1a,2B,3a)')
        else:
            print_sms('For more than one model (e.g: 1a,4B,7a)')
        print_sms('All models (e.g: all)')
        models_to_train = input('Type here the model(s):')
        incorrect_models, correct_models = check_models(models_to_train, selected_action)
    print(''.ljust(width_of_text, '-'))
    return selected_action, tuple(correct_models)


def check_models(text: str, action: int) -> tuple:
    """_summary_

    Args:
        text (_type_): str to check if all models are correct

    Returns:
        tuple: (incorrect name of  model(s), correct name of model(s) )
    """
    models_str = text.upper().split(',')
    if action == 1: # find best parameters
        max_quantity_models = 4
    elif action >= 2:   # Train and test models
        max_quantity_models = 7
    h=tuple(zip(range(1, max_quantity_models + 1), repeat('A', max_quantity_models)))
    k=tuple(zip(range(1, max_quantity_models + 1), repeat('B', max_quantity_models)))
    sc1 = [''.join((str(item[0]), item[1])) for item in h]
    sc2 = [''.join((str(item[0]), item[1])) for item in k]
    sc1.extend(sc2)
    sc1 = sorted(sc1)
    all_combinations = tuple(sc1)
    no_ok = list()
    if 'all' in text:
        models_ok = all_combinations
    else:
        models_ok = list()
        no_ok =list()
        for i,item in enumerate(models_str):
            if item.strip() in all_combinations:
                if action == 1:
                    models_ok.append(item.replace('4', '8'))  # rf is model number 8
                else:
                    models_ok.append(item)
            else:
                no_ok.append(i)
    return no_ok, models_ok
    
def print_sms(*kargs) -> None:
    """_summary_
    Format the message and print it
    Args:
        text (str): text to format the message
    """
    full_text = ''
    for item in kargs:
        full_text += f'{item}'
    full_text.upper().ljust(60, '-')
    result_to_txt.append(full_text)
    print(full_text)
    
def form_row(values: tuple) -> str:
    """_summary_

    Args:
        # model_name (str): Name of model
        # mod_range (str): range as str(e.g. 1-100)
        # best_lab (str): best estimator value as str for labeled data (e.g. '14')
        # best_orig(str): best estimator value as str for original data (e.g. '18')
        # max_leaf_lab(str): max number of leaf nodes for labeled data (e.g. '6')
        # max_leaf_orig(str): max number of leaf nodes for original data (e.g. '6')
    Returns:
        str: _description_
    """
    cs = 10
    # be = best_est
    # if best_est.isdigit():
    #     be = int(best_est)
    # x = '|' + f'{model_name}'.center(cs, ' ') + '|'  + f'{mod_range}'.center(cs, ' ') + '|' + f'{be}'.center(cs, ' ') + '|\n'
    model_name, mod_range_lab, best_lab, max_leaf_lab, mod_range_ori, best_orig, max_leaf_orig = values
    x= f"""|{model_name.center(cs, ' ')}|{mod_range_lab.center(cs, ' ')}|{best_lab.center(cs, ' ')}|{max_leaf_lab.center(cs, ' ')}|{mod_range_ori.center(cs, ' ')}|{best_orig.center(cs, ' ')}|{max_leaf_orig.center(cs, ' ')}|"""
    return x
    
def print_estimators_table():
    est_table = get_estimators_table()[-1]
    print(''.center(width_of_text - 3, '-'))
    cols_name = ('Model', 'RgLB', 'BELD', 'MLLD', 'RgOR', 'BEOD', 'MLOD')
    print(form_row(cols_name))
    est_table = get_estimators_table()[-1]
    tmp_list = list()
    print(''.center(width_of_text - 3, '-'))
    for item in (est_table.index):
        tmp_list = list(est_table.loc[item, :])
        tmp_list.insert(0, item)
        all_values = tmp_list
        all_values_str = tuple([str(val) for val in all_values])
        txt_print = form_row(all_values_str)
        print(txt_print)
    print(''.center(width_of_text - 3, '-'))
    print('RgLB --> Range for model(s) with labeled data')
    print('BELD --> Best estimator with labeled data')
    print('MLLD --> Max number of leaf nodes labeled data')
    print('RgLB --> Range for model(s) with original data')
    print('BEOD --> Best estimator with original data ')
    print('MLOD --> Max number of leaf nodes original data')
    
def train_and_test(models_to_run: tuple) -> None:
    for mod in models_to_run:
        model = models.get(int(mod[0]))
        if mod[1] == 'A':
            j = 1
        else:
            j = 2
        data_to_use = k_data.get(j)
        print_sms(f"Running model '{model}' with '{data_to_use}' dataset")
        dataset = process_data(data_to_use)
        print('The best estimator for each model was computed by GridSearchCV\n'
              'and are shown in following table:')
        print_estimators_table()
        trained_model = train_model(model, full_data=dataset, k_data=data_to_use)
        if not os.path.exists('models_saved'):
            os.mkdir('models_saved')
        joblib.dump(trained_model, f"models_saved/{mod}-{model}.pkl", compress=True)
        print_sms(f"the trained model have been saved as {mod}-{model}.pkl in the folder 'models_saved'")
        results, all_prediction = test_model(trained_model, dataset)
        print_sms("Report of model's performance")
        performance = get_model_performance(all_prediction_2022=all_prediction, full_data=dataset, all_results=results)
        result_to_txt.extend([str(item) for item in performance])
    with open('results.txt', 'w') as f:
        f.write('\n'.join(result_to_txt))
    print_sms("results was saved in in the file 'results.txt'")   
    print(''.ljust(width_of_text, '-'))
        
def find_best_estimator(models_to_run: tuple) -> None:     
    selected_range = ''
    for mod in models_to_run:
        if mod[1] == 'A':
            j = 1
        else:
            j = 2
        if mod[0] == 4:
            mod = '8' + mod[1]
        model = models.get(int(mod[0]))
        data_to_use = k_data.get(j)
        dataset = process_data(data_to_use)
        X = dataset.loc[:, '1_week_ago':'8_week_ago']
        estimators_table = get_estimators_table()[-1]
        if j == 1:
            y = dataset['status_clf']
        else:
            y = dataset['status']
        print_sms(f"Running model '{model}' with '{data_to_use}' dataset to find best estimator")
        dataset = process_data(data_to_use)
        while selected_range == '':
            if j==1:
                selected_range = estimators_table.loc[model, 'range_labeled']
            else:
                selected_range = estimators_table.loc[model, 'range_original']
            print_sms('The value of the parameters are:')
            print_estimators_table()
            print_sms('Please enter a range (e.g. 1-20) or leave it in blank')
            str_rng = input(f"to take range of model '{model}' from table: ") 
            if str_rng == '':
                print_sms(f'The selected range of {model} is:', selected_range)
            if str_rng.isdigit():
                selected_range = f'1-{str_rng}'
            if '-' in str_rng:
                vals = str_rng.split('-')
                ok_values = [True if item.isdigit() else False for item in vals]
                if False not in ok_values:
                    selected_range = f'{vals[0]}-{vals[1]}'
        if '-' in selected_range:
            model_range_str = selected_range.split('-')
        else:
            model_range_str = ['1', '100']
        model_range = tuple([int(item) for item in model_range_str])
        low_rg, up_rg = model_range
        param_grid = [
            {'n_estimators': tuple(range(low_rg, up_rg + 1))}
        ]
        rf_cl = RandomForestClassifier(random_state=42, max_leaf_nodes=8)        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        grid_SearCV = GridSearchCV(rf_cl, param_grid=param_grid, n_jobs=-1, verbose=1)
        input("hit ENTER")
        grid_SearCV.fit(X_train, y_train)
        best_est = grid_SearCV.best_estimator_.n_estimators
        max_leaf = grid_SearCV.best_estimator_.max_leaf_nodes
        print_sms('Best number of estimators:', best_est)
        # TODO Add range per kind of data
        # TODO Add option to enter dif range for each model
        if j==1:
            estimators_table.at[model, 'range_labeled'] = selected_range
            estimators_table.at[model, 'best_data_labeled'] = int(best_est)
            estimators_table.at[model, 'max_leaf_data_labeled'] = int(max_leaf)
        else:
            estimators_table.at[model, 'range_original'] = selected_range
            estimators_table.at[model, 'best_data_original'] = int(best_est)
            estimators_table.at[model, 'max_leaf_data_original'] = int(max_leaf)
        estimators_table.to_csv('best_estimators.csv')
        print_sms('Max number of leaf nodes:', max_leaf)
        print_sms('The Parameters have been saved')
        print(''.ljust(width_of_text, '-'))
        
        
def get_prediction():
    pass #TODO Prediction

if __name__ == '__main__':
    print_sms('Welcome to this toolbox')
    while True:
        selected_action, models_to_run = run_tool_box()
        print(models_to_run)
        if selected_action == 1:
            print_sms('RUNNING GRIDSEARCH TO FIND BEST PARAMETERS')
            find_best_estimator(models_to_run)
        elif selected_action == 3:
            get_prediction()
        else:
            print_sms('TRAINING AND TESTING MODELS')
            train_and_test(models_to_run)
    
     