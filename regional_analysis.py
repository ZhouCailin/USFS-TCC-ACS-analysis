"""
analyze the overview and regional disparity of the UTC data
cailin zhou
2 apr. 2023
"""

import os
import numpy as np
import pandas as pd
import inequalipy

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

result_df = pd.DataFrame(index = ["mean", "Pop.-weighted mean", "Pop.", "tracts", "Gini"], columns = list(regions_dict.keys()) + ["US"])

def to_numeric(df):
    varlist = [col for col in df.columns if col not in exclude_list]
    for col in varlist:
        df[col] = df[col].astype(str).str.replace(',', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')

def overview(df):
    # calculate mean of "MeanValue"
    mean = df["MeanValue"].mean()
    # gini
    gini = inequalipy.gini(df["MeanValue"])
    print("mean: ", mean)
    print("gini: ", gini)
    return mean, gini

def pop_weight_mean(df):
    # calculate the weight mean of MeanVal
    sum_pop = df["TotPop"].sum()
    mean = (df["MeanValue"] * (df["TotPop"] / sum_pop)).sum()
    print("pop-weighted mean: ", mean)
    return mean, sum_pop, df.shape[0]

def stat_by_zone(df):
    for zone in regions_dict:
        zone_df = df[df["State_y"].isin(regions_dict[zone])]
        print("zone: ", zone)
        overview(zone_df)
        pop_weight_mean(zone_df)
        result_df[zone]["mean"] = overview(zone_df)[0]
        result_df[zone]["Pop.-weighted mean"] = pop_weight_mean(zone_df)[0]
        result_df[zone]["Gini"] = overview(zone_df)[1]
        result_df[zone]["Pop."] = pop_weight_mean(zone_df)[1]
        result_df[zone]["tracts"] = pop_weight_mean(zone_df)[2]

def main():
    # ascfile here must have "totalPop" and "geoid" columns
    acsfile = workdir + '\\dexter_2309106228_extract.csv'
    uctresult = workdir + '\\tree_cover_tract_data.pkl'
    acs = pd.read_csv(acsfile, header=0, encoding="ISO-8859-1")
    utc = pd.read_pickle(uctresult)
    merged = pd.merge(acs, utc, left_on='geoid', right_on='NormID', how='inner')
    to_numeric(merged)
    stat_by_zone(merged)

    result_df["US"]["mean"] = overview(merged)[0]
    result_df["US"]["Pop.-weighted mean"] = pop_weight_mean(merged)[0]
    result_df["US"]["Gini"] = overview(merged)[1]
    result_df["US"]["Pop."] = pop_weight_mean(merged)[1]
    result_df["US"]["tracts"] = pop_weight_mean(merged)[2]

    result_df.to_csv(workdir + "\\regional_analysis_result.csv")

if __name__ == '__main__':
    main()