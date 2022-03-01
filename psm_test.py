# -*- coding: utf-8 -*-
import threading

import pandas as pd
import numpy as np
import patsy
import sys
from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.api as sm
# import statsmodels.regression.linear_model as sm
from plot_smd import plot_psm
from tqdm import tqdm
# import sys  # 导入sys模块
# sys.setrecursionlimit(10000)  # 将默认的递归深度修改为3000

#


def psm_pipei(data, X_field, Y_field, output_path, control_treated_dict):

    # 数据介绍
    # data 患者的数据
    # X_field 选择匹配的患者特征
    # Y_field 组别
    # output_psth 输出的文件位置与名称
    # 注意患者的分组列，0为对照组，1为实验组，即0多1少

    # 由于逻辑回归第三方包不支持某些特殊字符，故对特征列名称进行替换
    id_X_field_dict = {f'feature_{i}': X_field[i] for i in range(len(X_field))}
    X_field_id_dict = {X_field[i]: f'feature_{i}' for i in range(len(X_field))}
    data = data.rename(columns=X_field_id_dict)
    X_field = [name for name in id_X_field_dict]

    treated = data[data['group'] == 1]
    control = data[data['group'] == 0]
    y_f, x_f = patsy.dmatrices('{} ~ {}'.format(Y_field[0], '+'.join(X_field)), data=data, return_type='dataframe')
    formula = '{} ~ {}'.format(Y_field[0], '+'.join(X_field))
    # print('Formula:\n' + formula)
    # print('n majority:', len(control))
    # print('n minority:', len(treated))  # 确定回归方程

    # ---------------------------计算逻辑回归模型的准确率----------------------------
    ############################################################################

    i = 0
    nmodels = 5  # 可指定归回模型个数
    errors = 0
    model_accuracy = []
    models = []
    # global res

    while i < nmodels and errors < 5:
        sys.stdout.write('\r{}: {}\{}'.format("Fitting Models on Balanced Samples", i, nmodels))  # 第几个模型
        y_samp, X_samp = patsy.dmatrices(formula, data=data, return_type='dataframe')  # 选出模型的自变量和因变量
        glm = GLM(y_samp, X_samp, family=sm.families.Binomial())  # 逻辑回归模型
        try:
            # global res
            res = glm.fit()
            preds = [1.0 if i >= .5 else 0.0 for i in res.predict(X_samp)]
            preds = pd.DataFrame(preds)
            preds.columns = y_samp.columns
            b = y_samp.reset_index(drop=True)
            a = preds.reset_index(drop=True)
            ab_score = \
            ((a.sort_index().sort_index(axis=1) == b.sort_index().sort_index(axis=1)).sum() * 1.0 / len(y_samp)).values[0]  # 模型预测准确性得分
            model_accuracy.append(ab_score)
            models.append(res)
            i += 1
        except Exception as e:
            errors += 1
            print('Error: {}'.format(e))
    print("\nAverage Accuracy:", "{}%".format(round(np.mean(model_accuracy) * 100, 2)))  # 所有模型的平均准确性
    # -----------------计算倾向评分------------------------------
    preds = [i for i in res.predict(X_samp)]
    preds = pd.DataFrame(preds)
    preds.columns = ['scores']
    data = pd.concat([data, preds], axis=1)

    # 将特征列id名转换回特征原始名称
    data = data.rename(columns=id_X_field_dict)
    X_field = [name for name in X_field_id_dict]
    # ---------------------匹配-------------------
    threshold = 0.001
    method = 'min'
    nmatches = 1  # 匹配比，一个实验组患者匹配nmatch个对照组
    test_scores = data[data[Y_field[0]] == True][['scores']]
    ctrl_scores = data[data[Y_field[0]] == False][['scores']]
    result, match_ids = [], []
    for i in range(len(test_scores)):
        score = test_scores.iloc[i]
        matches = abs(ctrl_scores - score).sort_values('scores').head(nmatches)
        chosen = np.random.choice(matches.index, nmatches, replace=False)
        result.extend([test_scores.index[i]] + list(chosen))
        match_ids.extend([i] * (len(chosen) + 1))
        ctrl_scores = ctrl_scores.drop(chosen, axis=0)

    matched_data = data.loc[result]
    matched_data['match_id'] = match_ids
    matched_data['record_id'] = matched_data.index
    output_df = matched_data[['record_id', 'group']].rename(columns={'record_id': 'PATIENT_CODE'})
    output_df['PATIENT_CODE'] = output_df['PATIENT_CODE'].apply(lambda x: data.loc[x, 'PATIENT_CODE'])
    output_df['group'] = output_df['group'].apply(lambda x: control_treated_dict['control'] if x == 0 else x)
    output_df['group'] = output_df['group'].apply(lambda x: control_treated_dict['treated'] if x == 1 else x)
    output_df.to_excel(output_path, index=False)

def psm(input_norm, control_treated_dict, n, output_path, item_dict='-'):


    #
    if n <= 0:
        print('Error!')
        return False
    data = pd.read_excel(input_norm).dropna()  # 去除任何包含缺失数据的行
    data['group'] = data['group'].apply(lambda x: 0 if str(x) == control_treated_dict['control'] else x)
    data['group'] = data['group'].apply(lambda x: 1 if str(x) == control_treated_dict['treated'] else x)
    data = data[data['group'].isin([0, 1])]
    X_field = data.columns.tolist()
    X_field.remove('group')
    X_field.remove('PATIENT_CODE')
    Y_field = ['group']
    ####################数据预处理###########################################################################################
    # 数据清洗
    for i in tqdm(range(n)):
        output_pipei_path = output_path + f'matched_id_group_{i}.xlsx'
        data = data.sample(frac=1).reset_index(drop=True)
        psm_pipei(data, X_field, Y_field, output_pipei_path, control_treated_dict)
    plot_psm(input_norm, output_path, n, control_treated_dict, item_dict)


#####################输入内容###############################################################################################

bp = 'C:/Users/admin/Desktop/倾向性匹配/'

input_norm = bp + "psm.xlsx"  # 写具体的引用文件名，该文件数据包括所有的X变量和Y变量(组别特征)
output_path = bp + 'output/'

control_treated_dict = {"control": '分组2', "treated": '分组1'}

n = 1  # 匹配的次数
# 只输出中文版图片及结果
psm(input_norm, control_treated_dict, n, output_path)


# 增加输出英文版图片
# item_dict = {"住院时间": "time", "年龄": "age", "乏力": "test1", "腹胀": "test2", "口干": "test3", "纳差": "test4"}
# psm(input_norm, control_treated_dict, n, output_path, item_dict)

