import os
import numpy as np
import pandas as pd
import inequalipy.kolmpollak as kpindex
import inequalipy.atkinson as atkindex
import inequalipy

import seaborn as sns
import matplotlib.pyplot as plt

workdir = os.path.dirname(os.path.abspath(__file__))

regions_dict = {"Northeast": ['09', '10', '23', '24', '25', '33', '34', '36', '42', '44', '50'],
                "Upper Mideast": ['19', '26', '27', '55'],
                "Ohio Valley": ['17', '18', '21', '29', '39', '47', '54'],
                "Southeast": ['01', '12', '13', '37', '45', '51'],
                "Northern Rockies and Plains": ['30', '31', '38', '46', '56'],
                "South": ['05', '20', '22', '28', '40', '48'],
                "Southwest":['04', '08', '35', '49'],
                "Northwest": ['16', '41', '53'],
                "West": ['06', '32']}

exclude_list = ["geoid", "State_x", "County", "GEOID", "NormID", "State_y"]


def process(df):
    all_result = {}

    # to numeric for all columns
    varlist = [col for col in df.columns if col not in exclude_list]
    for col in varlist:
        df[col] = df[col].astype(str).str.replace(',', '')
        df[col] = df[col].astype(str).str.replace('$', '')
        df[col] = df[col].astype(str).str.replace(' ', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # generate new column
    df["POC"] = df["TotPop"] - df["NonHispWhite"]

    # df["NonHispOther"] = df["NonHispPop"] - df["NonHispWhite"] - df["NonHispBlack"]
    # df["Underpov"] = df["PovRatioUnderHalf"] + df["PovRatiov5tov99"]
    # df["LowIncome"] = df["HHInc0"] + df["HHInc10"] + df["HHInc15"]
    # df["MidIncome"] = df["HHInc25"] + df["HHInc35"] + df["HHInc50"] + df["HHInc75"]
    # df["HighIncome"] = df["HHInc100"] + df["HHInc150"] + df["HHInc200"]
    # df["LowFamIncome"] = df["FamHHInc0"] + df["FamHHInc10"] + df["FamHHInc15"]
    # df["MidFamIncome"] = df["FamHHInc25"] + df["FamHHInc35"] + df["FamHHInc50"] + df["FamHHInc75"]
    # df["HighFamIncome"] = df["FamHHInc100"] + df["FamHHInc150"] + df["FamHHInc200"]

    df["NotBach"] = df["TotPop"] - df["Under18"] - df["Bachelorsormore"]
    df["BornUS"] = df["TotPop"] - df["BornOutsideUS"]
    df["CanEng"] = df["TotPop"] - df["OthLangEnglishLTD"]
    df["USCitizen"] = df["TotPop"] - df["NonCitizen"]
    df["UnderPov"] = df["PovRatioUnderHalf"] + df["PovRatiov5tov99"]
    df["Otherage"] = df["TotPop"] - df["Under18"] - df["Over65"]

    varlist = [col for col in df.columns if col not in exclude_list]

    tract_wei_zone_result = pd.DataFrame(index=varlist)

    # subdf for each zone
    state_code_col = 'State_y'
    for region, states in regions_dict.items():
        zone_df = df[df['State_y'].isin(states)]
        zone_result = zone_process(zone_df, region)

        # calculate sum of each column
        fract_df = zone_df[varlist]/zone_df[varlist].sum()
        wei_result = fract_df.multiply(zone_df["MeanValue"], axis=0)
        tract_wei_zone_result[region] = wei_result.sum()

        all_result[region] = zone_result
        zone_result.to_csv(workdir+region+".csv")
        tract_wei_zone_result.to_csv(workdir+"\\utctract_wei_zone_result.csv")
    return all_result, varlist

def zone_process(zone_df, zone_name):
    # generate subdf for each state
    varlist = [col for col in zone_df.columns if col not in exclude_list]
    zone_result = pd.DataFrame(index=varlist)
    for state in regions_dict[zone_name]:
        state_df = zone_df[zone_df['State_y'] == state]
        state_result = state_process(state_df)
        zone_result = pd.concat([zone_result, state_result], axis=1)
    return zone_result

def state_process(state_df):
    varlist = [col for col in state_df.columns if col not in exclude_list]
    state_result = pd.DataFrame(index=varlist)
    # generate subdf for each county
    county_name_col = 'County'
    for county in state_df[county_name_col].unique():
        county_df = state_df[state_df[county_name_col] == county]
        county_result = county_process(county_df)
        state_result = pd.concat([state_result, county_result], axis=1)
    return state_result

def county_process(county_df):
    # result
    varlist = [col for col in county_df.columns if col not in exclude_list]
    result = pd.DataFrame(index=varlist, columns=['MeanValue'])

    # generate subdf for urban tracts
    # define urban tract: tract with population density > 2000
    pop_density_threshold = 0
    pop_col = 'TotPop'
    area_col = 'AreaSqMi'
    # county_urban_df = county_df [(county_df[pop_col] / county_df[area_col]) > pop_density_threshold]
    county_urban_df = county_df

    # calculate the weight mean of MeanVal
    for var in varlist:
        sum_pop = county_urban_df[var].sum()
        var_mean = (county_urban_df["MeanValue"] * (county_urban_df[var] / sum_pop)).sum()
        result["MeanValue"][var] = var_mean
    return result

def kp_inequality(X, kappa=1):
    """
    Calculate Kp inequality index for a given numpy array and kappa value.
    :param X: 1-D numpy array
    :param kappa: float, the inequality aversion parameter
    :return: float, the Kp inequality index
    """
    N = len(X)
    X.sort()
    W = np.exp(kappa * X)
    W_sum = np.sum(W)
    Z = np.cumsum(W) / W_sum
    Kp = -(1 / kappa) * np.log(1 / N * np.sum((np.exp(kappa * X) - 1) / W))
    return Kp

def postprocess(all_result, varlist):
    # calculate the Kp index for each zone
    mean_df = pd.DataFrame(columns=all_result.keys(), index=varlist)
    kp_df = pd.DataFrame(columns=all_result.keys(), index=varlist)
    atk_df = pd.DataFrame(columns=all_result.keys(), index=varlist)
    gini_df = pd.DataFrame(columns=all_result.keys(), index=varlist)
    for zone, zone_result in all_result.items():
        for index, row in zone_result.iterrows():
            data = row[row > 0]
            data = np.array(data.to_list())
            mean = data.mean()
            kp = kp_inequality(data)
            # kp = kpindex.index(data, kappa=0.5)
            atk = atkindex.index(data)
            gini = inequalipy.gini(data)
            gini_df[zone][index] = gini
            atk_df[zone][index] = atk
            mean_df[zone][index] = mean
            kp_df[zone][index] = kp


    # save to csv
    # mean_df.to_csv(workdir + '\\mean_all_other.csv')
    # kp_df.to_csv(workdir + '\\kp_all2.csv')
    # atk_df.to_csv(workdir + '\\atk_all2.csv')
    # gini_df.to_csv(workdir + '\\gini_all2.csv')


def main():
    acsfile = workdir + '\\dexter_2309206201_extract.csv'
    resultfile = workdir + '\\tree_cover_tract_data.pkl'
    acs = pd.read_csv(acsfile, header=0, encoding="ISO-8859-1")
    utc = pd.read_pickle(resultfile)
    merged = pd.merge(acs, utc, left_on='geoid', right_on='NormID', how='inner')
    all_result, varlist = process(merged)
    postprocess(all_result, varlist)

if __name__ == '__main__':
    main()
