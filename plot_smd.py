import PythonMeta as PMA
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *
import numpy as np


def showstudies(studies, dtype):
    #show continuous data
    if dtype.upper()=="CONT":
        text = "%-10s %-30s %-30s \n"%("Study ID","Experiment Group","Control Group")
        text += "%-10s %-10s %-10s %-10s %-10s %-10s %-10s \n"%(" ","m1","sd1","n1","m2","sd2","n2")
        for i in range(len(studies)):
            text += "%-10s %-10s %-10s %-10s %-10s  %-10s %-10s \n"%(
            studies[i][6],        # study ID
            str(studies[i][0]),   # mean of group1
            str(studies[i][1]),   # SD of group1
            str(studies[i][2]),   # total num of group1
            str(studies[i][3]),   # mean of group2
            str(studies[i][4]),   # SD of group2
            str(studies[i][5])    # total num of group2
            )
        return text

    #show dichotomous data
    text = "%-10s %-20s %-20s \n"%("Study ID","Experiment Group","Control Group")
    text += "%-10s %-10s %-10s %-10s %-10s \n"%(" ","e1","n1","e2","n2")
    for i in range(len(studies)):
        text += "%-10s %-10s %-10s %-10s %-10s \n"%(
        studies[i][4],        # study ID
        str(studies[i][0]),   # event num of group1
        str(studies[i][1]),   # total num of group1
        str(studies[i][2]),   # event num of group2
        str(studies[i][3])    # total num of group2
        )
    return text

def showresults(rults):
    text = "%-10s,%-6s,%-18s,%-10s" % ("Study ID", "n", "ES[95% CI]", "Weight(%)\n")
    for i in range(1, len(rults)):
        text += "%-10s,%-6d,%-4.2f,%.2f,%.2f,%6.2f\n" % (   # for each study
        rults[i][0],     #study ID
        rults[i][5],     #total num
        rults[i][1],     #effect size
        rults[i][3],     #lower of CI
        rults[i][4],     #higher of CI
        100*(rults[i][2]/rults[0][2])  #weight
        )
    text += "%-10s %-6d  %-4.2f[%.2f %.2f]   %6d\n"%(         # for total effect
        rults[0][0],     #total effect size name
        rults[0][5],     #total N (all studies)
        rults[0][1],     #total effect size
        rults[0][3],     #total lower CI
        rults[0][4],     #total higher CI
        100
        )
    text += "%d studies included (N=%d)\n"%(len(rults)-1,rults[0][5])
    text += "Heterogeneity: Tau\u00b2=%.3f "%(rults[0][12]) if not rults[0][12]==None else "Heterogeneity: "
    text += "Q(Chisquare)=%.2f(p=%s); I\u00b2=%s\n"%(
        rults[0][7],     #Q test value
        rults[0][8],     #p value for Q test
        str(round(rults[0][9],2))+"%")   #I-square value
    text += "Overall effect test: z=%.2f, p=%s\n"%(rults[0][10],rults[0][11])  #z-test value and p-value

    return text

def SMD_main(stys,settings,out_path):

    d = PMA.Data()  #Load Data class
    m = PMA.Meta()  #Load Meta class

    #You should always tell the datatype first!!!
    d.datatype = settings["datatype"]                #set data type, 'CATE' for binary data or 'CONT' for continuous data
    studies = d.getdata(stys)                        #load data
    #studies = d.getdata(d.readfile("studies.txt"))  #get data from a data file, see examples of data files
    text00 = showstudies(studies, d.datatype)
    # print(text00)           #show studies

    m.datatype=d.datatype                            #set data type for meta-analysis calculating
    m.models = settings["models"]                    #set effect models: 'Fixed' or 'Random'
    m.algorithm = settings["algorithm"]              #set algorithm, based on datatype and effect size
    m.effect = settings["effect"]                    #set effect size:RR/OR/RD for binary data; SMD/MD for continuous data
    results = m.meta(studies)                        #performing the analysis
    # print(studies)
    # print(results)
    # print(m.models + " " + m.algorithm + " " + m.effect)
    text01 = showresults(results)
    # oddsratio(18, 20050,205,377054)
    with open(out_path, 'w', encoding = 'utf8') as fw:
        # f.write('# '+data + '\n')
        # print(out_path)
        fw.write(text00 + '\n')
        fw.write(text01 + '\n')

    # print(text01)                     #show results table
    return showresults(results)
def my_main(stys_matched, stys_unmatched, settings):
    d = PMA.Data()  #Load Data class
    m = PMA.Meta()  #Load Meta class
    f = PMA.Fig()   #Load Fig class


    #You should always tell the datatype first!!!
    d.datatype = settings["datatype"]                #set data type, 'CATE' for binary data or 'CONT' for continuous data
    studies = d.getdata(stys_matched)                        #load data
    #studies = d.getdata(d.readfile("studies.txt"))  #get data from a data file, see examples of data files
    # print(showstudies(studies, d.datatype))           #show studies

    m.datatype=d.datatype                            #set data type for meta-analysis calculating
    m.models = settings["models"]                    #set effect models: 'Fixed' or 'Random'
    m.algorithm = settings["algorithm"]              #set algorithm, based on datatype and effect size
    m.effect = settings["effect"]                    #set effect size:RR/OR/RD for binary data; SMD/MD for continuous data
    results = m.meta(studies)                        #performing the analysis
    # print(m.models + " " + m.algorithm + " " + m.effect)
    # print (showresults(results))                     #show results table
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    f.forest(results).show()                         #show forest plot
    # f.funnel(results).show()                         #show funnel plot

    d.datatype = settings["datatype"]                #set data type, 'CATE' for binary data or 'CONT' for continuous data
    studies = d.getdata(stys_unmatched)                        #load data
    #studies = d.getdata(d.readfile("studies.txt"))  #get data from a data file, see examples of data files
    # print(showstudies(studies, d.datatype))           #show studies

    m.datatype=d.datatype                            #set data type for meta-analysis calculating
    m.models = settings["models"]                    #set effect models: 'Fixed' or 'Random'
    m.algorithm = settings["algorithm"]              #set algorithm, based on datatype and effect size
    m.effect = settings["effect"]                    #set effect size:RR/OR/RD for binary data; SMD/MD for continuous data
    results = m.meta(studies)                        #performing the analysis
    # print(m.models + " " + m.algorithm + " " + m.effect)
    # print (showresults(results))                     #show results table
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    f.forest(results).show()                         #show forest plot
    # f.funnel(results).show()                         #show funnel plot


def basic_unmatched(input_norm_df):

    result_list=[]
    item_list = input_norm_df.columns.tolist()
    item_list.remove('group')
    item_list.remove('PATIENT_CODE')

    for name in item_list:
        ### df1 所有实验组的特征df
        ### df2 所有控制组的特征df
        df1 = input_norm_df[input_norm_df['group'] == 1].copy()
        df2 = input_norm_df[input_norm_df['group'] == 0].copy()
        df1 = df1[name].dropna()
        df2 = df2[name].dropna()
        me1 = df1.mean()
        sd1 = df1.std()
        n1 = len(df1)
        me2 = df2.mean()
        sd2 = df2.std()
        n2 = len(df2)
        str1 = name+','+str(me1)+','+str(sd1)+','+str(n1)+','+str(me2)+','+str(sd2)+','+str(n2)
        result_list.append(str1)
    return result_list
def basic_matched(input_norm_df,id_df):
    treat_ID_list = []
    control_ID_list = []
    input_norm_df = pd.merge(input_norm_df, id_df, on='PATIENT_CODE', how='inner')
    result_list = []
    item_list = input_norm_df.columns.tolist()
    item_list.remove('group')
    item_list.remove('PATIENT_CODE')

    for name in item_list:
        ### df1 所有实验组的特征df
        ### df2 所有控制组的特征df
        df1 = input_norm_df[input_norm_df['group'] == 1].copy()
        df2 = input_norm_df[input_norm_df['group'] == 0].copy()
        df1 = df1[name].dropna()
        df2 = df2[name].dropna()
        me1 = df1.mean()
        sd1 = df1.std()
        n1 = len(df1)
        # print(treat_me1,treat_sd1,treat_n1)
        me2 = df2.mean()
        sd2 = df2.std()
        n2 = len(df2)
        str1= name+','+str(me1)+','+str(sd1)+','+str(n1)+','+str(me2)+','+str(sd2)+','+str(n2)
        result_list.append(str1)
    return result_list
def get_meta(input_norm_df, id_df, out_path_unmatched, out_path_matched, output_meta_path, settings):
    # 对数据进行meta分析，输出分析结果
    # df患者数据
    # matched_file 患者分组数据
    # out_path_unmatched 匹配前的smd结果
    # out_path_matched 匹配后的smd结果
    # output_meta_path meta分析结果
    # output_meta_path_e 转化为英文版的最终meta分析结果
    # output_meta_path和output_meta_path_e都可以用于plt_meta

    result_list_unmatched = basic_unmatched(input_norm_df)
    result_list_matched = basic_matched(input_norm_df, id_df)
    result = SMD_main(result_list_unmatched, settings, out_path_unmatched)
    a = result.split('\n')
    a = a[1:-5]
    a = '\n'.join(a)
    a = ' '.join(a.split(','))
    a = a.split()
    # fw = open(bp+'shuju/psm_data/plt/unmatched_meta.txt', 'w')
    # fw.write(a)
    list = []
    list = [[a[i-5], 'Unmatched cohort', float(a[i-4]), float(a[i-3]), float(a[i-2]), float(a[i-1]), float(a[i]), float(a[i-1])-float(a[i-2])] for i in range(len(a)) if (i+1)%6 == 0]
    a_df = pd.DataFrame(list, columns=['variable', 'Type', 'n', 'Standardized Mean Didderence', 'min', 'max', 'weight', 'max-min'])
    # a_df.to_csv(bp+'shuju/psm_data/plt/unmatched_meta.txt', index=False, sep='\t', encoding='gbk')
    result = SMD_main(result_list_matched, settings, out_path_matched)
    b = result.split('\n')
    b = b[1:-5]
    b = '\n'.join(b)
    b = ' '.join(b.split(','))
    b = b.split()
    # fw = open('unmatched_meta.txt', 'w')
    # fw.write(a)
    list = [[b[i-5], 'Matched cohort', float(b[i-4]), float(b[i-3]), float(b[i-2]), float(b[i-1]), float(b[i]), float(b[i-1])-float(b[i-2])] for i in range(len(b)) if (i+1)%6 == 0]
    b_df = pd.DataFrame(list, columns=['variable', 'Type', 'n', 'Standardized Mean Didderence', 'min',
                                'max', 'weight', 'max-min'])
    # b_df.to_csv('matched_meta.txt',index=False,sep='\t',encoding='gbk')
    ab_df = pd.concat([a_df, b_df], axis=0)
    ab_df.to_excel(output_meta_path, index=False)
def transform_meta_english(meta_path, meta_path_e, item_dict):
    # 将中文的meta分析结果的中文特征转化为英文
    meta_df = pd.read_excel(meta_path)
    meta_list = [x for x in item_dict]
    meta_df = meta_df[meta_df['variable'].isin(meta_list)]
    meta_df['variable'] = meta_df['variable'].apply(lambda x: x if item_dict[x] == '-' else item_dict[x])
    meta_df.to_excel(meta_path_e, index=False)
def plot_meta(meta_path, plt_path_1, plt_path_2, text_family='serif', text_size=12, x_text="Standardized Mean Difference", y_text="",
              margin_width=225, margin_height=150, matgin_units='mm'):
    # text_family 文字的字体格式
    # text_size 文字的大小
    # x_text 横坐标的标签
    # y_text 纵坐标的标签


    b = pd.read_excel(meta_path)
    ## Plot using ggplot

    p1 = (
            ggplot(data=b, mapping=aes(x='variable', y='Standardized Mean Didderence', group='Type', color='Type')) +
            geom_point() +
            geom_line(mapping=aes(x='variable', y='Standardized Mean Didderence')) +
            geom_hline(yintercept=0.1, color="black", size=0.1) +
            geom_hline(yintercept=-0.1, color="black", size=0.1) +
            geom_hline(yintercept=0.1, color="black", size=0.1) +
            geom_hline(yintercept=-0.1, color="black", size=0.1) +
            theme_bw() +
            coord_flip() +
            theme(legend_key=element_blank()) +

            labs(x=y_text, y=x_text, col='') +
            scale_y_continuous(breaks=np.arange(-1, 1, 0.1)) +
            theme(text=element_text(family=text_family)) +
            theme(axis_text_y=element_text(size=text_size, family=text_family, face="bold"),
                  axis_text_x=element_text(size=text_size, family=text_family, face="bold"))

    )
    p1.save(filename=plt_path_1, width=margin_width, height=margin_height, units=matgin_units)

    p2=(ggplot(data=b, mapping=aes(x='Standardized Mean Didderence', y='variable', group='Type', color='Type')) +
       geom_point() +
       geom_vline(xintercept=0.1, color="black", size=0.1) +
       geom_vline(xintercept=-0.1, color="black", size=0.1) +
       geom_errorbarh(mapping=aes(y='variable', xmax='max', xmin='min'), height=0.5) +theme_bw() +
       theme(legend_key=element_blank()) +
       theme(panel_grid_major=element_blank(), panel_grid_minor=element_blank(),
             panel_background=element_blank()) +
       labs(x=x_text, y=y_text, col='') +
       scale_x_continuous(breaks=np.arange(-1, 1, 0.1)) +
       theme(text=element_text(family=text_family)) +
       theme(axis_text_y=element_text(size=text_size, family=text_family, face="bold"),
             axis_text_x=element_text(size=text_size, family=text_family, face="bold"))
)
    p2.save(filename=plt_path_2, width=margin_width, height=margin_height, units=matgin_units)


def plot_psm(input_norm_path, output_path, n, control_treated_dict, item_dict):
    #sample: continuous data
    settings={"datatype":"CONT",  #for CONTinuous data
    "models":"Fixed",             #models: Fixed or Random
    "algorithm":"IV",             #algorithm: IV
    "effect":"SMD"}                #effect size: MD, SMD
    input_norm_df = pd.read_excel(input_norm_path)
    input_norm_df['group'] = input_norm_df['group'].apply(lambda x: 0 if str(x) == control_treated_dict['control'] else x)
    input_norm_df['group'] = input_norm_df['group'].apply(lambda x: 1 if str(x) == control_treated_dict['treated'] else x)
    input_norm_df = input_norm_df[input_norm_df['group'].isin([0, 1])]
    for i in range(n):
        matched_file = output_path + f'matched_id_group_{i}.xlsx'
        out_path_unmatched = output_path + f'smd_unmatched_{i}.txt'
        out_path_matched = output_path + f'smd_matched_{i}.txt'
        output_meta_path = output_path + f'meta_{i}.xlsx'
        output_meta_path_e = output_path + f'meta_英文版_{i}.xlsx'
        plt_path_1 = output_path + f'smd_plt1_{i}.png'
        plt_path_2 = output_path + f'smd_plt2_{i}.png'
        plt_path_1_e = output_path + f'smd_英文版_plt1_{i}.png'
        plt_path_2_e = output_path + f'smd_英文版_plt2_{i}.png'
        id_df = pd.read_excel(matched_file)[['PATIENT_CODE']]

        get_meta(input_norm_df, id_df, out_path_unmatched, out_path_matched, output_meta_path, settings)
        plot_meta(output_meta_path, plt_path_1, plt_path_2, text_family='Microsoft YaHei', text_size=8)   # 中文用Microsoft YaHei
        if item_dict != '-':
            transform_meta_english(output_meta_path, output_meta_path_e, item_dict)  # 进行特征的中英文转换
            plot_meta(output_meta_path_e, plt_path_1_e, plt_path_2_e, text_family='serif', text_size=5)  # 英文用setif