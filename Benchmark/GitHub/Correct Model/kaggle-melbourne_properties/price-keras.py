##!/usr/bin/python
import matplotlib
matplotlib.use('Agg')
import shutil
import pandas as pd
import numpy as np
from tensorflow.contrib import keras
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import argparse
import pickle
import os

class Swish(keras.layers.Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
        return (keras.activations.sigmoid(x)*x)

def string2integer(ref,data,key):
    if key in ref:
        select = data[key]
        integer_encoding = LabelEncoder().fit_transform(select)
        data[key] = integer_encoding


def guess_regionname(data,target_suburb,target_street):
    ref = data['Regionname'].isnull()
    # if there is entry within same subrn then return its region if available
    for i in range(0,len(data)):
        suburb = data['Suburb'][i]
        if target_suburb == suburb and ref[i] == False:
            return data['Regionname'][i]  
        
    # if there is entry within street name then return its RegionName
    for i in range(0,len(data)):
        street = data['Address'].values[i][data['Address'].values[i].find(" ")+1:]
        if street == target_street and ref[i] == False:
            return data['Regionname'][i]  
    return "Unknown"


def fill_region_name_up(data):
    ref = data['Regionname'].isnull()
    for i in range(0,len(ref)):
        if ref[i] == True:
            street = data['Address'].values[i][data['Address'].values[i].find(" ")+1:]
            suburb = data['Suburb'][i]
            data['Regionname'][i] = guess_regionname(data,suburb,street)
    return


def guess_property_count(data,target_suburb):
    ref = data['Propertycount'].isnull()
    for i in range(0,len(data)):
        if target_suburb == data['Suburb'][i] and ref[i] == False:
            return data['Propertycount'][i]
    return 1 

def fill_property_count_up(data):
    """ Idea is to find property count from other entries of the same suburb"""
    ref = data['Propertycount'].isnull()
    for i in range(0,len(ref)):
        if ref[i] == True:
            data['Propertycount'][i] = guess_property_count(data,data['Suburb'][i]) 
    return


def guess_distance(data, target_suburb):
    ref = data['Distance'].isnull()
    sum_distance = 0.0
    sum_counter = 0.0
    for i in range(0,len(data)):
         if ref[i] == False and data['Suburb'][i] == target_suburb:
            sum_distance += data['Distance'][i]
            sum_counter += 1.0
    if sum_counter == 0:
        sum_counter = 1.0
    return sum_distance/sum_counter


def fill_distance_up(data):
    """ Distance to C.B.D taken as an averge distance from other properties in
        the same suburb located on the same street. If no properties on the same street
        then averge distance in the same suburb"""
    ref = data['Distance'].isnull()
    to_drop = []
    for i in range(0,len(ref)):
        if ref[i] == True:
            data['Distance'][i] = guess_distance(data,data['Suburb'][i]) 
            # If in ths suburb there is only one property with borken data
            # then delete this entry
            if data['Distance'][i] == 0:
                to_drop.append(i)
    return data.drop(to_drop)
    

def guess_yearbuilt(data, target_suburb, target_type):
    ref = data['YearBuilt'].isnull()
    sum_yearbuilt = 0.0
    sum_counter = 0.0
    for i in range(0,len(data)):
         if (ref[i] == False) and (data['Suburb'][i] == target_suburb) and (data['Type'][i] == target_type):
            sum_yearbuilt += data['YearBuilt'][i]
            sum_counter += 1.0
    if sum_counter == 0:
        sum_counter = 1.0
    return sum_yearbuilt/sum_counter


def fill_yearbuilt_up(data):
    """ If year build is missing find same type property from same suburb and takes its year 
    provided it is no fresher that Date of transaction"""
    ref = data['YearBuilt'].isnull()
    for i in range(0,len(ref)):
        if ref[i] == True:
            data['YearBuilt'][i] = guess_yearbuilt(data,data['Suburb'][i],data['Type'][i])
            print("Type: %s Suburb: %s Guessed year: %d" %(data['Type'][i],data['Suburb'][i],data['YearBuilt'][i]))
    return 

def guess_value_by_interpolation(ref,data,target_suburb,target_type,target_key,target_val, gap_key, properties_stats):
    if target_val == pd.np.nan or target_val == 0:
        return 0
    neighbour_less_stats = properties_stats[(properties_stats['Suburb'] == target_suburb) &
            (properties_stats[target_key] <= target_val)]
    neighbour_greater_stats = properties_stats[(properties_stats['Suburb'] == target_suburb) &
            (properties_stats[target_key] > target_val)]
    if len(neighbour_less_stats) != 0 and len(neighbour_greater_stats) != 0:
        min_index = neighbour_less_stats[target_key].argmax()
        max_index = neighbour_greater_stats[target_key].argmin()
        delta_y = neighbour_greater_stats[gap_key][max_index] - neighbour_less_stats[gap_key][min_index] 
        delta_x = neighbour_greater_stats[target_key][max_index] - neighbour_less_stats[target_key][min_index] 
        estimate = (target_val - neighbour_less_stats[target_key][min_index]) * delta_y/delta_x + neighbour_less_stats[gap_key][min_index]
    elif len(neighbour_less_stats) != 0:
        min_index = neighbour_less_stats[target_key].argmax()
        estimate = neighbour_less_stats[gap_key][min_index]
    elif len(neighbour_greater_stats) !=0:
        max_index = neighbour_greater_stats[target_key].argmin()
        estimate = neighbour_greater_stats[gap_key][max_index]
    else:
        estimate = 0
    return estimate 

def fill_buildingarea_up(data):
    """ Estimate BuildingArea based on type of property and LandSize is available """

    ref = data['BuildingArea'].isnull()
    # Make a hash table ['type' : data]
    properties_stats = {}
    properties_stats['h'] = data[(data['Type'] == "h") & (data['Landsize'].notnull())
            & (data['BuildingArea'].notnull())]
    properties_stats['t'] = data[(data['Type'] == "t") & (data['Landsize'].notnull())
            & (data['BuildingArea'].notnull())]
    properties_stats['br'] = data[(data['Type'] == "br") & (data['Landsize'].notnull())
            & (data['BuildingArea'].notnull())]
    properties_stats['u'] = data[(data['Type'] == "u") & (data['Landsize'].notnull())
            & (data['BuildingArea'].notnull())]
    properties_stats['dev site'] = data[(data['Type'] == "dev site") & (data['Landsize'].notnull())
            & (data['BuildingArea'].notnull())]
    properties_stats['o res'] = data[(data['Type'] == "o res") & (data['Landsize'].notnull())
            & (data['BuildingArea'].notnull())]
    
    for i in range(0,len(ref)):
        if ref[i] == True:
            data['BuildingArea'][i] = guess_value_by_interpolation(ref,data,data['Suburb'][i],data['Type'][i],
                    'Landsize',data['Landsize'][i], 'BuildingArea', properties_stats[data['Type'][i]])
    return data


def fill_landsize_up(data):
    """ Estimate LAndsize based on type of property and BuildingArea if available """

    ref = data['Landsize'].isnull()
    # Make a hash table ['type' : data]
    properties_stats = {}
    properties_stats['h'] = data[(data['Type'] == "h") & (data['BuildingArea'].notnull())
            & (data['Landsize'].notnull())]
    properties_stats['t'] = data[(data['Type'] == "t") & (data['BuildingArea'].notnull())
            & (data['Landsize'].notnull())]
    properties_stats['br'] = data[(data['Type'] == "br") & (data['BuildingArea'].notnull())
            & (data['Landsize'].notnull())]
    properties_stats['u'] = data[(data['Type'] == "u") & (data['BuildingArea'].notnull())
            & (data['Landsize'].notnull())]
    properties_stats['dev site'] = data[(data['Type'] == "dev site") & (data['BuildingArea'].notnull())
            & (data['Landsize'].notnull())]
    properties_stats['o res'] = data[(data['Type'] == "o res") & (data['BuildingArea'].notnull())
            & (data['Landsize'].notnull())]

    for i in range(0,len(ref)):
        if ref[i] == True:  # guess building area
            data['Landsize'][i] = guess_value_by_interpolation(ref,data,data['Suburb'][i],data['Type'][i],
                    'BuildingArea',data['BuildingArea'][i], 'Landsize', properties_stats[data['Type'][i]])
    return data


def fill_alley_up(data):

    """ Alley value of NA means no alley access to property 
    Instead of NA we put 0, and Grvl = 1 , Pave = 2"""

    data['Alley'].fillna(0,inplace=True)
    data['Alley'].replace("Grvl",1,inplace=True)
    data['Alley'].replace("Pave",2,inplace=True)
    return


def fill_lotfrontage(data):

    data['LotFrontage'].fillna(0,inplace=True)

#    plt.plot(data["LotFrontage"].values,data["SalePrice"].values,'ro')
#    plt.title("Lot Frontage Influence")
#    plt.xlabel("Lot Frontage")
#    plt.ylabel("SoldPrice")
#    plt.legend(["SoldPrice"],loc="upper right")
#    plt.show()
#    plt.savefig("lotfrontage.png")

    return


def fill_fireplaceqa_up(data):

    """ Alley value of NA means no alley access to property 
    Instead of NA we put 0, and Grvl = 1 , Pave = 2"""

    data['FireplaceQu'].fillna(0,inplace=True)
    data['FireplaceQu'].replace("Po",1,inplace=True)
    data['FireplaceQu'].replace("Fa",2,inplace=True)
    data['FireplaceQu'].replace("TA",3,inplace=True)
    data['FireplaceQu'].replace("Gd",4,inplace=True)
    data['FireplaceQu'].replace("Ex",5,inplace=True)
    return


def fill_garageyearbuild_up(data):
    """ Year when garage was build is to be discarded"""
    data.drop(['GarageYrBlt'], axis=1, inplace=True )  # Drop column with Garage Year build
    return 

def fill_garagetype_up(data):
    """ N/A means no garage. Other values are text and will be converted to integers"""

    data['GarageType'].fillna(0,inplace=True)
    data['GarageType'].replace("Detchd",1,inplace=True)
    data['GarageType'].replace("CarPort",2,inplace=True)
    data['GarageType'].replace("BuiltIn",3,inplace=True)
    data['GarageType'].replace("Basment",4,inplace=True)
    data['GarageType'].replace("Attchd",5,inplace=True)
    data['GarageType'].replace("2Types",6,inplace=True)

    return


def fill_garagequal_up(data):
    data['GarageQual'].fillna(0,inplace=True)
    data['GarageQual'].replace("Po",1,inplace=True)
    data['GarageQual'].replace("Fa",2,inplace=True)
    data['GarageQual'].replace("TA",3,inplace=True)
    data['GarageQual'].replace("Gd",4,inplace=True)
    data['GarageQual'].replace("Ex",5,inplace=True)

    return 

def fill_garagecars_up(data):
    data['GarageCars'].fillna(0,inplace=True)
    return

def fill_garagearea_up(data):
    data['GarageArea'].fillna(0,inplace=True)
    return

def fill_garagecond_up(data):
    data['GarageCond'].fillna(0,inplace=True)
    data['GarageCond'].replace("Po",1,inplace=True)
    data['GarageCond'].replace("Fa",2,inplace=True)
    data['GarageCond'].replace("TA",3,inplace=True)
    data['GarageCond'].replace("Gd",4,inplace=True)
    data['GarageCond'].replace("Ex",5,inplace=True)

    return 


def fill_garagefinish_up(data):
    """ state of interior finish"""

    data['GarageFinish'].fillna(0,inplace=True)
    data['GarageFinish'].replace("Unf",1,inplace=True)
    data['GarageFinish'].replace("RFn",2,inplace=True)
    data['GarageFinish'].replace("Fin",3,inplace=True)

    return


def fill_poolqc_up(data):
    data['PoolQC'].fillna(0,inplace=True)
    data['PoolQC'].replace("Fa",1,inplace=True)
    data['PoolQC'].replace("TA",2,inplace=True)
    data['PoolQC'].replace("Gd",3,inplace=True)
    data['PoolQC'].replace("Ex",4,inplace=True)

    return 


def fill_fence_up(data):
    data['Fence'].fillna(0,inplace=True)
    data['Fence'].replace("MnWw",1,inplace=True)
    data['Fence'].replace("GdWo",2,inplace=True)
    data['Fence'].replace("MnPrv",3,inplace=True)
    data['Fence'].replace("GdPrv",4,inplace=True)

    return 

# Consider one hot representation
def fill_misc_up(data):
    data['MiscFeature'].fillna(0,inplace=True)
    data['MiscFeature'].replace("Elev",1,inplace=True)
    data['MiscFeature'].replace("Gar2",2,inplace=True)
    data['MiscFeature'].replace("TenC",3,inplace=True)
    data['MiscFeature'].replace("Shed",4,inplace=True)
    data['MiscFeature'].replace("Othr",5,inplace=True)
    return


def fill_bsmtqual_up(data):

    data['BsmtQual'].fillna(0,inplace=True)
    data['BsmtQual'].replace("Po",1,inplace=True)
    data['BsmtQual'].replace("Fa",2,inplace=True)
    data['BsmtQual'].replace("TA",3,inplace=True)
    data['BsmtQual'].replace("Gd",4,inplace=True)
    data['BsmtQual'].replace("Ex",5,inplace=True)
    return


def fill_bsmtcond_up(data):

    data['BsmtCond'].fillna(0,inplace=True)
    data['BsmtCond'].replace("Po",1,inplace=True)
    data['BsmtCond'].replace("Fa",2,inplace=True)
    data['BsmtCond'].replace("TA",3,inplace=True)
    data['BsmtCond'].replace("Gd",4,inplace=True)
    data['BsmtCond'].replace("Ex",5,inplace=True)
    return


def fill_bsmtexposure_up(data):

    data['BsmtExposure'].fillna(0,inplace=True)
    data['BsmtExposure'].replace("No",1,inplace=True)
    data['BsmtExposure'].replace("Mn",2,inplace=True)
    data['BsmtExposure'].replace("Av",3,inplace=True)
    data['BsmtExposure'].replace("Gd",4,inplace=True)
    return

def fill_bsmtfinsf1(data):
    data['BsmtFinSF1'].fillna(0,inplace=True)
    return

def fill_bsmtfinsf2(data):
    data['BsmtFinSF2'].fillna(0,inplace=True)
    return


def fill_bsmtunfsf(data):
    data['BsmtUnfSF'].fillna(0,inplace=True)
    return

def fill_bsmttotalsf(data):
    data['TotalBsmtSF'].fillna(0,inplace=True)
    return

def fill_bsmtfintype1_up(data):

    data['BsmtFinType1'].fillna(0,inplace=True)
    data['BsmtFinType1'].replace("Unf",1,inplace=True)
    data['BsmtFinType1'].replace("LwQ",2,inplace=True)
    data['BsmtFinType1'].replace("Rec",3,inplace=True)
    data['BsmtFinType1'].replace("BLQ",4,inplace=True)
    data['BsmtFinType1'].replace("ALQ",5,inplace=True)
    data['BsmtFinType1'].replace("GLQ",6,inplace=True)
    return


def fill_bsmtfullbath_up(data):
    # If none declared then assume 0
    data['BsmtFullBath'].fillna(0,inplace=True)
    return

def fill_bsmthalfbath_up(data):
    # If none declared then assume 0
    data['BsmtHalfBath'].fillna(0,inplace=True)
    return

def fill_bsmtfintype2_up(data):

    data['BsmtFinType2'].fillna(0,inplace=True)
    data['BsmtFinType2'].replace("Unf",1,inplace=True)
    data['BsmtFinType2'].replace("LwQ",2,inplace=True)
    data['BsmtFinType2'].replace("Rec",3,inplace=True)
    data['BsmtFinType2'].replace("BLQ",4,inplace=True)
    data['BsmtFinType2'].replace("ALQ",5,inplace=True)
    data['BsmtFinType2'].replace("GLQ",6,inplace=True)
    return

def fill_masvnrarea_up(data):
    # We treat N/A as lack of masonary vaneer area 
    data['MasVnrArea'].fillna(0,inplace=True)
    return

def fill_electrical_up(data):

    total = len(data['Electrical'].values)
    total_mix = len(data[data.Electrical == "Mix"].values)
    total_fusep = len(data[data.Electrical == "FuseP"].values)
    total_fusef = len(data[data.Electrical == "FuseF"].values)
    total_fusea = len(data[data.Electrical == "FuseA"].values)
    total_sbrk = len(data[data.Electrical == "SBrkr"].values)

    # Missing data is a building fairly new , so after
    # 1960s all have SBRKR type of electrial
    data['Electrical'].fillna("SBrkr",inplace=True)
    data['Electrical'].replace("Mix",1,inplace=True)
    data['Electrical'].replace("FuseP",2,inplace=True)
    data['Electrical'].replace("FuseF",3,inplace=True)
    data['Electrical'].replace("FuseA",4,inplace=True)
    data['Electrical'].replace("SBrkr",5,inplace=True)

    # turn into integers and plot to see correlations
    #candidate = "YearBuilt"
    #plt.plot(range(0,1456),data["Electrical"].values,'ro')
    #plt.plot(data[candidate],data["Electrical"].values,'ro')
    #plt.title("Electrical correlations")
    #plt.xlabel(candidate)
    #plt.ylabel("Electrical")
    #plt.legend(["Electrical"],loc="upper right")
    #plt.show()
    #plt.savefig("electrical.png")
    return

def convert_mszoning(data):

    #data.MSZoning[455] = "RM"
    #data.MSZoning[756] = "RM"
    #data.MSZoning[790] = "RM"
    #data.MSZoning[1444] = "RL"
    # Mitchel -> RL
    # IDOTRR -> RM

    for i in range(0,len(data)):
        if data.MSZoning.isnull()[i]:
            if data.Neighborhood[i] == "Mitchel":
                data.MSZoning[i] = "RL"
            if data.Neighborhood[i] == "IDOTRR":
                data.MSZoning[i] = "RM"

    categories = {'A' : 0, 'C': 1, "C (all)" : 2, "FV" : 3, "I" : 4, "RH" : 5, "RL" : 6, "RP" : 7, "RM" : 8}
    for key in categories.keys():
        data['MSZoning'].replace(key,categories[key],inplace=True)
    if args.onehot:
        onehots = []
        row = [ 0 for i in range(0,len(categories))]
        for entity in data["MSZoning"]:
            data_row = row
            data_row[entity] = 1
            onehots.append(data_row)

        df_onehots = pd.DataFrame(data=onehots, columns=["MSZoning_"+str(i) for i in range(0,len(categories))])
        data = pd.concat([data,df_onehots],axis=1)
        # Remove integer encoded column
        data = data.drop('MSZoning',axis=1)
    return data


def convert_street(data):

    # TODO: convert to one hot representation
    data['Street'].replace("Grvl",-1,inplace=True)
    data['Street'].replace("Pave",1,inplace=True)
    return


def convert_lotshape(data):

    data['LotShape'].replace("IR3",1,inplace=True)
    data['LotShape'].replace("IR2",2,inplace=True)
    data['LotShape'].replace("IR1",3,inplace=True)
    data['LotShape'].replace("Reg",4,inplace=True)
    return


def convert_LandContour(data):

    # TODO: Make one hot
    data['LandContour'].replace("Low",1,inplace=True)
    data['LandContour'].replace("HLS",2,inplace=True)
    data['LandContour'].replace("Bnk",3,inplace=True)
    data['LandContour'].replace("Lvl",4,inplace=True)
    return


def convert_Utilities(data):

    # TODO: Make one hot
    data['Utilities'].fillna("AllPub",inplace=True)
    data['Utilities'].replace("ELO",1,inplace=True)
    data['Utilities'].replace("NoSeWa",2,inplace=True)
    data['Utilities'].replace("NoSewr",3,inplace=True)
    data['Utilities'].replace("AllPub",4,inplace=True)
    return


def convert_lotconfig(data):
    # TODO: Make one hot
    data['LotConfig'].replace("Inside",1,inplace=True)
    data['LotConfig'].replace("Corner",2,inplace=True)
    data['LotConfig'].replace("CulDSac",3,inplace=True)
    data['LotConfig'].replace("FR2",4,inplace=True)
    data['LotConfig'].replace("FR3",5,inplace=True)
    return


def convert_LandSlope(data):

    data['LandSlope'].replace("Sev",1,inplace=True)
    data['LandSlope'].replace("Mod",2,inplace=True)
    data['LandSlope'].replace("Gtl",3,inplace=True)
    return

def convert_Neighbourhood(data):

    data['Neighborhood'].replace("Blmngtn",1,inplace=True)
    data['Neighborhood'].replace("Blueste",2,inplace=True)
    data['Neighborhood'].replace("BrDale",3,inplace=True)
    data['Neighborhood'].replace("BrkSide",4,inplace=True)
    data['Neighborhood'].replace("ClearCr",5,inplace=True)
    data['Neighborhood'].replace("CollgCr",6,inplace=True)
    data['Neighborhood'].replace("Crawfor",7,inplace=True)
    data['Neighborhood'].replace("Edwards",8,inplace=True)
    data['Neighborhood'].replace("Gilbert",9,inplace=True)
    data['Neighborhood'].replace("Greens",10,inplace=True)
    data['Neighborhood'].replace("GrnHill",11,inplace=True)
    data['Neighborhood'].replace("IDOTRR",12,inplace=True)
    data['Neighborhood'].replace("Landmrk",13,inplace=True)
    data['Neighborhood'].replace("MeadowV",14,inplace=True)
    data['Neighborhood'].replace("Mitchel",15,inplace=True)
    data['Neighborhood'].replace("NAmes",16,inplace=True)
    data['Neighborhood'].replace("NoRidge",17,inplace=True)
    data['Neighborhood'].replace("NPkVill",18,inplace=True)
    data['Neighborhood'].replace("NridgHt",19,inplace=True)
    data['Neighborhood'].replace("NWAmes",20,inplace=True)
    data['Neighborhood'].replace("OldTown",21,inplace=True)
    data['Neighborhood'].replace("SWISU",22,inplace=True)
    data['Neighborhood'].replace("Sawyer",23,inplace=True)
    data['Neighborhood'].replace("SawyerW",24,inplace=True)
    data['Neighborhood'].replace("Somerst",25,inplace=True)
    data['Neighborhood'].replace("StoneBr",26,inplace=True)
    data['Neighborhood'].replace("Timber",27,inplace=True)
    data['Neighborhood'].replace("Veenker",28,inplace=True)

    return


def convert_Condition(data):

    data['Condition1'].replace("Artery",1,inplace=True)
    data['Condition1'].replace("Feedr",2,inplace=True)
    data['Condition1'].replace("Norm",3,inplace=True)
    data['Condition1'].replace("RRNn",4,inplace=True)
    data['Condition1'].replace("RRAn",5,inplace=True)
    data['Condition1'].replace("PosN",6,inplace=True)
    data['Condition1'].replace("PosA",7,inplace=True)
    data['Condition1'].replace("RRNe",8,inplace=True)
    data['Condition1'].replace("RRAe",9,inplace=True)

    data['Condition2'].replace("Artery",1,inplace=True)
    data['Condition2'].replace("Feedr",2,inplace=True)
    data['Condition2'].replace("Norm",3,inplace=True)
    data['Condition2'].replace("RRNn",4,inplace=True)
    data['Condition2'].replace("RRAn",5,inplace=True)
    data['Condition2'].replace("PosN",6,inplace=True)
    data['Condition2'].replace("PosA",7,inplace=True)
    data['Condition2'].replace("RRNe",8,inplace=True)
    data['Condition2'].replace("RRAe",9,inplace=True)
    return


def convert_BldgType(data):

    data['BldgType'].replace("1Fam",6,inplace=True)
    data['BldgType'].replace("2FmCon",5,inplace=True)
    data['BldgType'].replace("2fmCon",5,inplace=True)
    data['BldgType'].replace("Duplx",4,inplace=True)
    data['BldgType'].replace("Duplex",4,inplace=True)
    data['BldgType'].replace("Twnhs",3,inplace=True)
    data['BldgType'].replace("TwnhsE",2,inplace=True)
    data['BldgType'].replace("TwnhsI",1,inplace=True)
    return



def convert_HouseStyle(data):

    data['HouseStyle'].replace("1Story",1,inplace=True)
    data['HouseStyle'].replace("1.5Fin",2,inplace=True)
    data['HouseStyle'].replace("1.5Unf",3,inplace=True)
    data['HouseStyle'].replace("2Story",4,inplace=True)
    data['HouseStyle'].replace("2.5Fin",5,inplace=True)
    data['HouseStyle'].replace("2.5Unf",6,inplace=True)
    data['HouseStyle'].replace("SFoyer",7,inplace=True)
    data['HouseStyle'].replace("SLvl",8,inplace=True)
    return


def convert_RoofStyle(data):

    data['RoofStyle'].replace("Flat",1,inplace=True)
    data['RoofStyle'].replace("Gable",2,inplace=True)
    data['RoofStyle'].replace("Gambrel",3,inplace=True)
    data['RoofStyle'].replace("Hip",4,inplace=True)
    data['RoofStyle'].replace("Mansard",5,inplace=True)
    data['RoofStyle'].replace("Shed",6,inplace=True)
    return


def convert_RoofMatl(data):

    data['RoofMatl'].replace("ClyTile",1,inplace=True)
    data['RoofMatl'].replace("CompShg",2,inplace=True)
    data['RoofMatl'].replace("Membran",3,inplace=True)
    data['RoofMatl'].replace("Metal",4,inplace=True)
    data['RoofMatl'].replace("Roll",5,inplace=True)
    data['RoofMatl'].replace("Tar&Grv",6,inplace=True)
    data['RoofMatl'].replace("WdShake",7,inplace=True)
    data['RoofMatl'].replace("WdShngl",8,inplace=True)
    return


def convert_Exterior(data):

    data['Exterior1st'].replace("AsbShng",1,inplace=True)
    data['Exterior1st'].replace("AsphShn",2,inplace=True)
    data['Exterior1st'].replace("BrkComm",3,inplace=True)
    data['Exterior1st'].replace("Brk Cmn",3,inplace=True)
    data['Exterior1st'].replace("BrkFace",4,inplace=True)
    data['Exterior1st'].replace("CBlock",5,inplace=True)
    data['Exterior1st'].replace("CemntBd",6,inplace=True)
    data['Exterior1st'].replace("CmentBd",6,inplace=True)
    data['Exterior1st'].replace("HdBoard",7,inplace=True)
    data['Exterior1st'].replace("ImStucc",8,inplace=True)
    data['Exterior1st'].replace("MetalSd",9,inplace=True)
    data['Exterior1st'].fillna("Other",inplace=True)
    data['Exterior1st'].replace("Other",10,inplace=True)
    data['Exterior1st'].replace("Plywood",11,inplace=True)
    data['Exterior1st'].replace("PreCast",12,inplace=True)
    data['Exterior1st'].replace("Stone",13,inplace=True)
    data['Exterior1st'].replace("Stucco",14,inplace=True)
    data['Exterior1st'].replace("VinylSd",15,inplace=True)
    data['Exterior1st'].replace("Wd Sdng",16,inplace=True)
    data['Exterior1st'].replace("WdShing",17,inplace=True)
    data['Exterior1st'].replace("Wd Shng",18,inplace=True)
       	
    data['Exterior2nd'].replace("AsbShng",1,inplace=True)
    data['Exterior2nd'].replace("AsphShn",2,inplace=True)
    data['Exterior2nd'].replace("BrkComm",3,inplace=True)
    data['Exterior2nd'].replace("Brk Cmn",3,inplace=True)
    data['Exterior2nd'].replace("BrkFace",4,inplace=True)
    data['Exterior2nd'].replace("CBlock",5,inplace=True)
    data['Exterior2nd'].replace("CemntBd",6,inplace=True)
    data['Exterior2nd'].replace("CmentBd",6,inplace=True)
    data['Exterior2nd'].replace("HdBoard",7,inplace=True)
    data['Exterior2nd'].replace("ImStucc",8,inplace=True)
    data['Exterior2nd'].replace("MetalSd",9,inplace=True)
    data['Exterior2nd'].fillna("Other",inplace=True)
    data['Exterior2nd'].replace("Other",10,inplace=True)
    data['Exterior2nd'].replace("Plywood",11,inplace=True)
    data['Exterior2nd'].replace("PreCast",12,inplace=True)
    data['Exterior2nd'].replace("Stone",13,inplace=True)
    data['Exterior2nd'].replace("Stucco",14,inplace=True)
    data['Exterior2nd'].replace("VinylSd",15,inplace=True)
    data['Exterior2nd'].replace("Wd Sdng",16,inplace=True)
    data['Exterior2nd'].replace("WdShing",17,inplace=True)
    data['Exterior2nd'].replace("Wd Shng",18,inplace=True)
    return


def convert_MasVnrType(data):
    data['MasVnrType'].replace("BrkCmn",1,inplace=True)
    data['MasVnrType'].replace("Brk Cmn",1,inplace=True)
    data['MasVnrType'].replace("BrkFace",2,inplace=True)
    data['MasVnrType'].replace("CBlock",3,inplace=True)
    data['MasVnrType'].replace("None",4,inplace=True)
    data['MasVnrType'].fillna(4,inplace=True)
    data['MasVnrType'].replace("Stone",5,inplace=True)
    return


def convert_ExterQual(data):

    data['ExterQual'].replace("Ex",5,inplace=True)
    data['ExterQual'].replace("Gd",4,inplace=True)
    data['ExterQual'].replace("TA",3,inplace=True)
    data['ExterQual'].replace("Fa",2,inplace=True)
    data['ExterQual'].replace("Po",1,inplace=True)
    return


def convert_ExterCond(data):

    data['ExterCond'].replace("Ex",5,inplace=True)
    data['ExterCond'].replace("Gd",4,inplace=True)
    data['ExterCond'].replace("TA",3,inplace=True)
    data['ExterCond'].replace("Fa",2,inplace=True)
    data['ExterCond'].replace("Po",1,inplace=True)
    return

def convert_Foundation(data):
    data['Foundation'].replace("BrkTil",6,inplace=True)
    data['Foundation'].replace("CBlock",5,inplace=True)
    data['Foundation'].replace("PConc",4,inplace=True)
    data['Foundation'].replace("Slab",3,inplace=True)
    data['Foundation'].replace("Stone",2,inplace=True)
    data['Foundation'].replace("Wood",1,inplace=True)
    return


def convert_Heating(data):
    data['Heating'].replace("Floor",6,inplace=True)
    data['Heating'].replace("GasA",5,inplace=True)
    data['Heating'].replace("GasW",4,inplace=True)
    data['Heating'].replace("Grav",3,inplace=True)
    data['Heating'].replace("OthW",2,inplace=True)
    data['Heating'].replace("Wall",1,inplace=True)
    return

def convert_HeatingQC(data):
    data['HeatingQC'].replace("Ex",5,inplace=True)
    data['HeatingQC'].replace("Gd",4,inplace=True)
    data['HeatingQC'].replace("TA",3,inplace=True)
    data['HeatingQC'].replace("Fa",2,inplace=True)
    data['HeatingQC'].replace("Po",1,inplace=True)
    return


def convert_Functional(data):

    data['Functional'].fillna("Typ",inplace=True)
    data['Functional'].replace("Typ",8,inplace=True)
    data['Functional'].replace("Min1",7,inplace=True)
    data['Functional'].replace("Min2",6,inplace=True)
    data['Functional'].replace("Mod",5,inplace=True)
    data['Functional'].replace("Maj1",4,inplace=True)
    data['Functional'].replace("Maj2",3,inplace=True)
    data['Functional'].replace("Sev",2,inplace=True)
    data['Functional'].replace("Sal",1,inplace=True)
    return


def convert_KitchenQual(data):

    data['KitchenQual'].replace("Ex",5,inplace=True)
    data['KitchenQual'].replace("Gd",4,inplace=True)
    data['KitchenQual'].fillna("TA",inplace=True)
    data['KitchenQual'].replace("TA",3,inplace=True)
    data['KitchenQual'].replace("Fa",2,inplace=True)
    data['KitchenQual'].replace("Po",1,inplace=True)
    return


def convert_CentralAir(data):
    data['CentralAir'].replace("N",0,inplace=True)
    data['CentralAir'].replace("Y",1,inplace=True)
    return

def convert_SaleType(data):
    data['SaleType'].replace("WD",10,inplace=True)
    data['SaleType'].replace("CWD",9,inplace=True)
    data['SaleType'].replace("VWD",8,inplace=True)
    data['SaleType'].replace("New",7,inplace=True)
    data['SaleType'].replace("COD",6,inplace=True)
    data['SaleType'].replace("Con",5,inplace=True)
    data['SaleType'].replace("ConLw",4,inplace=True)
    data['SaleType'].replace("ConLI",3,inplace=True)
    data['SaleType'].replace("ConLD",2,inplace=True)
    data['SaleType'].replace("Oth",1,inplace=True)
    return


def convert_SaleCondition(data):
    data['SaleCondition'].replace("Normal",6,inplace=True)
    data['SaleCondition'].replace("Abnorml",5,inplace=True)
    data['SaleCondition'].replace("AdjLand",4,inplace=True)
    data['SaleCondition'].replace("Alloca",3,inplace=True)
    data['SaleCondition'].replace("Family",2,inplace=True)
    data['SaleCondition'].replace("Partial",1,inplace=True)
    return



def convert_PavedDrive(data):
    data['PavedDrive'].replace("N",0,inplace=True)
    data['PavedDrive'].replace("P",1,inplace=True)
    data['PavedDrive'].replace("Y",2,inplace=True)
    return

def convert_GrLivArea(data):

    # Based on paper on IOWA dataset and chart GRLivArea/SalePrice
    # We can see that SalePrice of houses GrLivArea of values above 4000 square feets are strange
    # paper recommented to remove them so we did
    # Author of Iowa data set recommend to remove
    # bigger GrLivArea that 4000 , but it cannot be done
    # as kaggle submission (test) require also that one
    # We remove only in training mode
    # data = data[data.GrLivArea <= 4000]

    # For training, remove 
    to_drop = []
    if args.train == True:
        for i in range(0,len(data)):
            if data.GrLivArea[i] >= 4000 and data.SalePrice[i] < 300000 :
                to_drop.append(i)
    data.drop(to_drop,inplace=True)
    return data

def load_and_preprocess_comp_data(data_path):
    data = pd.read_csv(data_path)

    fill_alley_up(data)
    fill_lotfrontage(data)
    fill_fireplaceqa_up(data)
    fill_garagetype_up(data)
    fill_garageyearbuild_up(data)
    fill_garagefinish_up(data)
    fill_garagequal_up(data)
    fill_garagecond_up(data)
    fill_garagecars_up(data)
    fill_garagearea_up(data)
    fill_poolqc_up(data)
    fill_fence_up(data)
    fill_misc_up(data)
    fill_bsmtqual_up(data)
    fill_bsmtcond_up(data)
    fill_bsmtexposure_up(data)
    fill_bsmtfintype1_up(data)
    fill_bsmtfintype2_up(data)
    fill_bsmtfinsf1(data)
    fill_bsmtfinsf2(data)
    fill_bsmtunfsf(data)
    fill_bsmttotalsf(data)
    fill_bsmtfullbath_up(data)
    fill_bsmthalfbath_up(data)
    fill_electrical_up(data)
    fill_masvnrarea_up(data)
    data = convert_mszoning(data)
    data = convert_GrLivArea(data)
    convert_street(data)
    convert_lotshape(data)
    convert_LandContour(data)
    convert_Utilities(data)
    convert_lotconfig(data)
    convert_LandSlope(data)
    convert_Neighbourhood(data)
    convert_Condition(data)
    convert_BldgType(data)
    convert_HouseStyle(data)
    convert_RoofStyle(data)
    convert_RoofMatl(data)

    convert_Exterior(data)
    convert_Exterior(data)
    convert_MasVnrType(data)
    convert_ExterQual(data)
    convert_ExterCond(data)
    convert_Foundation(data)
    convert_Heating(data)
    convert_HeatingQC(data)
    convert_Functional(data)
    convert_KitchenQual(data)
    convert_CentralAir(data)
    convert_SaleType(data)
    convert_SaleCondition(data)
    convert_PavedDrive(data)
    #if "SalePrice" in data:
    #    print(len(data["SalePrice"].values))
    return data

def load_and_preprocess_data(melbourne_file_path):
    melbourne_data = pd.read_csv(melbourne_file_path)
    print(melbourne_data.isnull().sum()) # This is printing missing data
    # Car: number of parking spots. NAN will be replaced wth zeros
    melbourne_data['Car'].fillna(0,inplace=True)
    # Bathroom: number of bathrroms. NAN will be replaced with ones
    melbourne_data['Bathroom'].fillna(1,inplace=True)
    # Try to make up the field of council area
    melbourne_data['CouncilArea'].fillna("Unknown",inplace=True)
    # Try to fill the missing RegionName
    fill_region_name_up(melbourne_data)
    # Try to fill the missing PropertyCount (properites count in suburb)
    fill_property_count_up(melbourne_data)
    # Try to guess/even out YearBuild
    melbourne_data['YearBuilt'].fillna(0,inplace=True)
    # Building Area estimation 
    melbourne_data = fill_buildingarea_up(melbourne_data)
    # Landsize estimation
    melbourne_data = fill_landsize_up(melbourne_data)

    # Distance to C.B.D to be made up based on street+suburb info
    melbourne_data = fill_distance_up(melbourne_data)

    melbourne_data = melbourne_data.dropna(axis=0, subset = ['Price'])  # Drop data that contains NAN in Price column
    print(melbourne_data.isnull().sum()) # This is printing missing data
    if args.preprocess_only == True:
        melbourne_data.to_csv(args.dataset[5:].strip('.csv')+"_preprocessed.csv")
        return melbourne_data
    ref = melbourne_data.tail().to_dict()
    # Replace string values of CouncilArea with onehott representation
    string2integer(ref,melbourne_data,'CouncilArea')
    # Replace suburb name with integer encoding value
    string2integer(ref,melbourne_data,'Suburb')
    # Type of property
    string2integer(ref,melbourne_data,'Type')
    # enocode regionname 
    string2integer(ref,melbourne_data,'Regionname')
    # enocode SellerG 
    string2integer(ref,melbourne_data,'SellerG')
    # enocode Method 
    string2integer(ref,melbourne_data,'Method')
    # preprocess adress, remove apartmetn number , leaving just street
    if 'Address' in ref:
        address =  melbourne_data['Address']
        for i in range(0,len(address.values)):
            address.values[i] = address.values[i][address.values[i].find(" ")+1:]
        integer_encoding = LabelEncoder().fit_transform(address)
        melbourne_data['Address'] = integer_encoding

    # Day + month*32 + 32*12*year
    if 'Date' in ref:
        datas = melbourne_data['Date']
        for i in range(0,len(datas.values)):
            splits = datas.values[i].split('/')
            datas.values[i] = int(splits[0]) + int(splits[1])*32 + int(splits[2])*32*12
    return melbourne_data

def prepare_melbourne_dataset(melbourne_dataset_file):
    melbourne_data = load_and_preprocess_data(melbourne_dataset_file)
    # Ignored : Bedroom2, Latitude, Longtitude
    input_features = ['Address','Bathroom','BuildingArea','Car','CouncilArea', 'Date', 'Distance', 'Landsize', 'Method', 'Postcode', 'Price', 'Propertycount', 'Regionname', 'Rooms', 'SellerG', 'Suburb', 'Type', 'YearBuilt']
    return train_test_split(melbourne_data[input_features],melbourne_data.Price)

def prepare_competition_dataset(data_file):
    data = load_and_preprocess_comp_data(data_file)
    # Ignored:
    # Id,SalePrice, YrSold, MoSold, SaleType
    if args.onehot:
        input_features = ['Alley','MSSubClass',
            'MSZoning_0', 'MSZoning_1', 'MSZoning_2', 'MSZoning_3', 'MSZoning_4','MSZoning_5', 'MSZoning_6', 'MSZoning_7','MSZoning_8',
            'LotFrontage', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',  'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'SaleCondition']
    else:
        input_features = ['Alley','MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',  'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'SaleCondition']
    return data['Id'], data[input_features], data['SalePrice'] if args.train == True or args.validate == True else None

def parse_desc(model_desc):
    ''' Parse desc of model. Format is num of units in first layer, num of units in next layer..
        num of unit is last layer(has to be one)'''
    desc = []
    splitted = model_desc.split(',')
    for item in splitted:
        desc.append(int(item))
    if desc[len(desc)-1] != 1:
      print("Error: final layer should have one unit!")
      exit(-1)
    return desc


def make_model(model_name,model_desc,num_features):
    # Building model: 
    desc = parse_desc(model_desc)
    adam = keras.optimizers.Adam() # Does converge slowly 
    melbourne_model = keras.models.Sequential(name="-"+model_name + "-Adam")
    
    desc_str = "Tanh model: "
    for i in range(0,len(desc)):
        if i == 0:
            melbourne_model.add(keras.layers.Dense(desc[i], activation='tanh',
                kernel_regularizer=keras.regularizers.l1(0.0001) if args.regularize == True else None,
                kernel_initializer='glorot_normal', input_dim=num_features))
            desc_str += str(desc[i])
        elif desc[i] == 1:
            melbourne_model.add(keras.layers.Dense(desc[i], activation='linear', 
                kernel_regularizer=keras.regularizers.l1(0.0001) if args.regularize == True else None,
                kernel_initializer='glorot_normal'))
            desc_str += "-"+str(desc[i])
        else:
            melbourne_model.add(keras.layers.Dense(desc[i], activation='tanh', 
                kernel_regularizer=keras.regularizers.l1(0.0001) if args.regularize == True else None,
                kernel_initializer='glorot_normal'))
            desc_str += "-"+str(desc[i])

    print(desc_str)
    melbourne_model.compile(loss=args.loss, optimizer=adam) # Does converge slowly
    return melbourne_model

def make_relu_model(model_name,model_desc,num_features):

    # Prase description of model
    desc = parse_desc(model_desc)

    # Building model: 
    adam = keras.optimizers.Adam() # Does converge slowly 
    melbourne_model = keras.models.Sequential(name="-"+model_name + "-Adam")
    
    desc_str = "Relu model: "
    for i in range(0,len(desc)):
        if i == 0:
            melbourne_model.add(keras.layers.Dense(desc[i], activation='relu',
                kernel_regularizer=keras.regularizers.l1(0.0001) if args.regularize == True else None,
                kernel_initializer='he_normal', input_dim=num_features))
            desc_str += str(desc[i])
        else:
            melbourne_model.add(keras.layers.Dense(desc[i], activation='relu', 
                kernel_regularizer=keras.regularizers.l1(0.0001) if args.regularize == True else None,
                kernel_initializer='he_normal'))
            desc_str += "-"+str(desc[i])

    print(desc_str)
    melbourne_model.compile(loss=args.loss, optimizer=adam)
    return melbourne_model

def make_swish_model(model_name,model_desc,num_features):

    # Prase description of model
    desc = parse_desc(model_desc)

    # Building model: 
    adam = keras.optimizers.Adam() # Does converge slowly 
    melbourne_model = keras.models.Sequential(name="-"+model_name + "-Adam")
    keras.utils.get_custom_objects().update({'swish' : Swish(swish)})

    desc_str = "Swish model: "
    for i in range(0,len(desc)):
        if i == 0:
            melbourne_model.add(keras.layers.Dense(desc[i], activation='swish', 
                kernel_regularizer=keras.regularizers.l1(0.0001) if args.regularize == True else None,
                kernel_initializer='he_normal', input_dim=num_features))
            desc_str += str(desc[i])
        else:
            melbourne_model.add(keras.layers.Dense(desc[i], activation='swish' ,
                kernel_regularizer=keras.regularizers.l1(0.0001) if args.regularize == True else None,
                kernel_initializer='he_normal'))
            desc_str += "-"+str(desc[i])

    print(desc_str)
    melbourne_model.compile(loss=args.loss, optimizer=adam)
    return melbourne_model


def make_selu_model(model_name,model_desc,num_features):

    # Prase description of model
    desc = parse_desc(model_desc)

    desc_str = "SNN model: "
    melbourne_model = keras.models.Sequential(name="-"+model_name+"-Adam")

    for i in range(0,len(desc)):
        if i == 0:
            melbourne_model.add(keras.layers.Dense(desc[i], kernel_initializer='lecun_normal',
                activation='selu',input_dim=num_features))
            desc_str += str(desc[i])
        else:
            melbourne_model.add(keras.layers.Dense(desc[i], kernel_initializer='lecun_normal',
                activation='selu'))   # MAE: 
            desc_str += "-"+str(desc[i])

    print(desc_str)

    adam = keras.optimizers.Adam() # Does converge slowly 
    #sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9) # MAE: 
    melbourne_model.compile(loss=args.loss, optimizer=adam) # Does converge slowly
    return melbourne_model

def normalize_input(output_dir, data):
    """ Modify data so it is zero meaned """
    # Get all data from selected columns across samples
    # TODO: remove warnings
    list_of_scalers = {}
    if args.scaler != "":
        scalers = open(args.scaler,"rb")
        list_of_scalers = pickle.load(scalers)
    i = 0
    for col in data:
        if args.scaler == "":
            scaler = StandardScaler().fit(data[col].values.reshape(-1,1))
            list_of_scalers[col] = scaler 
        else:
            scaler = list_of_scalers[col]
        data[col] = scaler.transform([data[col]]).flatten()
    if args.scaler == "" and output_dir != "":
        scalers = open(output_dir + "/scalers", "wb")
        pickle.dump(list_of_scalers, scalers)
    return

def train(model_name, model_desc, num_epochs, X, y):

    initial_epoch = 1
    if model_name == "":
        model_name = "FFN"
    output = model_name + "_" + model_desc + "-" + args.loss + "-num_epochs-" + str(num_epochs)
    # TODO: Get name of model from compiled model and make
    # name of file to save to based on that
    if os.path.isdir(output):
        shutil.rmtree(output)
    os.mkdir(output)
    if args.normalize == True:
        normalize_input(output,X)

    if model_name == "" or model_name == "FFN": 
        melbourne_model = make_relu_model(model_name,model_desc,len(X.columns))
    elif model_name == "SNN":
        melbourne_model = make_selu_model(model_name,model_desc,len(X.columns))
    elif model_name == "SFN": # Swish forward network
        melbourne_model = make_swish_model(model_name,model_desc,len(X.columns))
    elif model_name == "TFN": # Tanh (traditional) forward network
        melbourne_model = make_model(model_name,model_desc,len(X.columns))
    else:
        keras.utils.get_custom_objects().update({'swish' : Swish(swish)})
        tmpstr = model_name[:model_name.find("-val_loss")]
        initial_epoch = int(tmpstr[tmpstr.rfind("-")+1:])
        melbourne_model = keras.models.load_model(model_name,custom_objects={'swish' : Swish(swish)})
    num_epochs += initial_epoch - 1

    show_stopper = keras.callbacks.EarlyStopping(monitor='val_loss',patience=num_epochs-10, verbose=1)
    print("Model learning params: %d" %(melbourne_model.count_params()))
    if args.checkpoints == True:
        checkpoint = keras.callbacks.ModelCheckpoint(filepath=output+"/melbourne_model"+melbourne_model.name+".epoch-{epoch:02d}-val_loss-{val_loss:.4f}.hdf5",monitor='val_loss',save_best_only=True,verbose=1)
        history = melbourne_model.fit(X.values, y.values, validation_split=0.2, epochs=num_epochs, initial_epoch=initial_epoch, batch_size=1,callbacks=[show_stopper,checkpoint])
    else:
        history = melbourne_model.fit(X.values, y.values, validation_split=0.2, epochs=num_epochs, initial_epoch=initial_epoch, batch_size=1,callbacks=[show_stopper])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Loss/cost chart")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend(["train","val"],loc="upper right")
    plt.show()
    plt.savefig(output+"/melbourne_model"+melbourne_model.name+"-epochs-"+str(num_epochs))
    print("Making predictions for following houses")
    predictions = melbourne_model.predict(X.values)

    print("MAE:",mean_squared_log_error(predictions,y.values))
# The Melbourne data has somemissing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.  
# Your Iowa data doesn't have missing values in the predictors you use. 
# So we will take the simplest option for now, and drop those houses from our data. 
#Don't worry about this much for now, though the code is:

def validate(model_name, X, Y):
    if model_name == "":
        print("Error: Validation mode require model given with --model option")
        exit(-1)
    melbourne_model = keras.models.load_model(model_name, custom_objects={'swish' : Swish(swish)})
    if args.scaler != "":
        normalize_input("",X)
    predictions = melbourne_model.predict(X.values)
    print("MAE:",mean_squared_log_error(predictions,Y.values))
    return

def infer(model_name, X, Ids):
    if model_name == "":
        print("Error: Inference mode require model given with --model option")
        exit(-1)

    melbourne_model = keras.models.load_model(model_name, custom_objects={'swish' : Swish(swish)})
    if args.scaler != "":
        normalize_input("",X)
    predictions = melbourne_model.predict(X.values)
    print("Id,SalePrice")        
    for i in range(0,len(Ids)):
        print("%d,%f" %(Ids.values[i],predictions[i]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Perform training", action="store_true")
    parser.add_argument("--preprocess_only", help="Perform data cleaning and store results in a file", action="store_true")
    parser.add_argument("--infer", help="Perform evaluation", action="store_true")
    parser.add_argument("--validate", help="Perform validation check", action="store_true")
    parser.add_argument("--type", help="Type of Model to be used for training/inference", type=str, default="")
    parser.add_argument("--model", help="Model to be used for training/inference", type=str, default="20,20,1")
    parser.add_argument("--scaler", help="Use standard scaler instead of computing one from data", type=str, default="")
    parser.add_argument("--loss", help="Loss function to be used", type=str, default="mean_squared_error")
    parser.add_argument("--dataset", help="Data Set for training/inference", type=str, default="comp:train.csv")
    parser.add_argument("--num_epochs", help="Number of epochs to perform", type=int, default=10)
    parser.add_argument("--normalize", help="Perform normalization", action="store_true")
    parser.add_argument("--regularize", help="Perform regularization L1", action="store_true")
    parser.add_argument("--checkpoints", help="Perform storing model params on checkpoints", action="store_true")
    parser.add_argument("--onehot", help="Perform one hot encoding for arbitrary selected columns", action="store_true")
    args = parser.parse_args()
    
    if args.dataset[0:4] == 'melb':
        trainX, testX, trainY, testY = prepare_melbourne_dataset(args.dataset[5:])
    elif args.dataset[0:4] == 'comp':
        Ids, dataX, dataY  = prepare_competition_dataset(args.dataset[5:])
    else:
        print("Invalid value of dataset. Accepted values: 'comp' and 'melb'");

    if args.preprocess_only == True:
        pass
    elif args.train == True:
        train(args.type,args.model,args.num_epochs, dataX, dataY)
    elif args.infer == True:
        infer(args.model, dataX, Ids)
    elif args.validate == True:
        validate(args.model, dataX, dataY)
    else:
        print("Error: Please specify either train of infer commandline option")
        exit(1)
