# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rc('font', family = 'Malgun Gothic')
PATH = r'F:\06_데이터\dacon_전력\mrc\dacon_elec\png'


def dayCounsumption(df: pd.DataFrame, x):
    imsi = df.groupby(['building_number', 'month', 'day', 'building_type'])[['power']].sum()
    imsi.reset_index(inplace=True)
    a1 = imsi.groupby(['building_type', 'month', 'day'])[['power']].mean()
    a1.reset_index(inplace=True)
    a1['new_day'] = list(map(lambda x, y : '0' + str(x) + '-' + str(y) if len(str(y)) != 1 else '0' + str(x) + '-' + '0' + str(y), a1['month'], a1['day']))

    pivot_df = a1.pivot(index='new_day', columns='building_type', values='power')
    png_name = len(pivot_df.columns)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_ylim(0, x)
    sns.lineplot(data=pivot_df)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    
    plt.title('Mean Power Consumption by Day and Building Type')
    plt.xlabel('Day')
    plt.ylabel('Mean Power Consumption')
    plt.legend(title='Building Type', loc='upper left')
    
    plt.savefig(PATH + f'\\{png_name}Mean Power Consumption by Day and Building Type.png')
        
    return plt.show()

def typeHourConsumption(df: pd.DataFrame, bd_type: str):
    func_df = df.loc[df['building_type'] == bd_type, :]
    pivot_df = func_df.groupby(['hour', 'building_number'])['power'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.lineplot(data=pivot_df)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    
    bd_type = bd_type.upper()
    plt.title(f'{bd_type}\n\n Mean Power Consumption by Hour')
    plt.xlabel('Hour')
    plt.ylabel('Mean Power Consumption')
    plt.legend(title='Building Number', loc='upper left')
    
    plt.savefig(PATH + f'\\{bd_type} Mean Power Consumption by Hour.png')
    
    return plt.show()

