import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

workdir = os.path.dirname(os.path.abspath(__file__))

var_list = ["Density", "POCPct", "WhitePct", "BlackPct", "PoorPct", "RichPct",
            "BachelorPct", "BornOutsideUSPct", "OthLangEnglishLTD", 'AvgHHInc', "MedianHHInc",
            'AvgHHEarnings', 'AvgFamInc', "MedianFamInc",
            "Over65", "Under18"]

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

def by_zone(df, zone):
        zone_df = df[df["State_y"].isin(regions_dict[zone])]
        print("zone: ", zone)
        return zone_df

def to_numeric(df):
    varlist = [col for col in df.columns if col not in exclude_list]
    for col in varlist:
        df[col] = df[col].astype(str).str.replace(',', '')
        df[col] = df[col].astype(str).str.replace('$', '')
        df[col] = df[col].astype(str).str.replace(' ', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')

def preprocess(df):
    to_numeric(df)

    df["Density"] = df["TotPop"]/df["LandSqMi"]
    df["POCPct"] = (df["TotPop"] - df["NonHispWhite"])/df["TotPop"]
    df["WhitePct"] = df["NonHispWhite"]/df["TotPop"]
    df["BlackPct"] = df["NonHispBlack"]/df["TotPop"]
    df["PoorPct"] = (df["PovRatioUnderHalf"] + df["PovRatiov5tov99"])/ df["TotPop"]
    df["RichPct"] = (df["PovRatioOver2"])/ df["TotPop"]
    df["BachelorPct"] = df["Bachelorsormore"]/(df["TotPop"] - df["Under18"])
    df["BornOutsideUSPct"] = df["BornOutsideUS"]/df["TotPop"]
    df["OthLangEnglishLTD"] = df["OthLangEnglishLTD"]/df["TotPop"]

    df.fillna(df.mean(), inplace=True)  # fill NaN with column means  # fill inf with column means


    return df


def pearson_corr(df, y_col, alpha=0.05):
    # Calculate the Pearson correlation and p-value between each column and Y
    corr_list = []
    sig_cols = []
    for col in df.columns:
        if col != y_col:
            corr, p = stats.pearsonr(df[col], df[y_col])
            corr_list.append((col, corr, p))
            if p < alpha:
                sig_cols.append(col)

    # Print the correlation and p-value for each column
    print("Pearson Correlation and P-values:")
    for col, corr, p in corr_list:
        print(f"{col}: Corr={corr:.3f}, P={p:.3f}")

    # Return the list of significant columns
    return sig_cols


def stepwise_selection_adjr2(X, y,
                             initial_list=[],
                             threshold_in=0.01,
                             threshold_out=0.05,
                             verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_adjr2_max = -np.inf
        best_feature = None
        for feature in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [feature]])).fit()
            adjr2 = model.rsquared_adj
            if adjr2 > new_adjr2_max:
                new_adjr2_max = adjr2
                best_feature = feature
        if new_adjr2_max > threshold_in:
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with adjusted R-squared {:.6f}'.format(best_feature, new_adjr2_max))
        model = sm.OLS(y, sm.add_constant(X[included])).fit()
        p_values = model.pvalues.iloc[1:]
        worst_feature = p_values.idxmax()
        pmax = p_values.max()
        if pmax > threshold_out:
            included.remove(worst_feature)
            changed = True
            if verbose:
                print('Drop {:30} with adjusted R-squared {:.6f}'.format(worst_feature, model.rsquared_adj))
        if not changed:
            break
    return included

def backward_elimination_sklearn(X, y, sig_level=0.05):
    # Initialize the list of selected features
    selected_cols = list(X.columns)

    # Run the regression with all features
    model = LinearRegression().fit(X, y)

    # Loop through and remove features until none are significant at the given level
    while True:
        # Get the p-values of each feature
        p_values = model.coef_

        # Find the highest p-value above the significance level
        max_p_value = np.max(np.abs(p_values))
        if max_p_value > sig_level:
            # Remove the feature with the highest p-value
            remove_feature = X.columns[np.argmax(np.abs(p_values))]
            selected_cols.remove(remove_feature)

            # Refit the model with the remaining features
            model = LinearRegression().fit(X[selected_cols], y)
        else:
            break

    # Print the final equation and variables
    print('Best model equation:')
    print('y = {:.6f}'.format(model.intercept_), end=' ')
    for i, col in enumerate(selected_cols):
        print('{:.6f} * {} +'.format(model.coef_[i], col), end=' ')
    print('')

    print('Variables in the best model:')
    print(selected_cols)

    return selected_cols

# Perform stepwise regression
def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=0.01,
                       threshold_out = 0.05,
                       verbose=True):
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


def backward_elimination(X, y, sig_level=0.05):
    # Initialize the list of selected features
    selected_cols = list(X.columns)

    # Run the regression with all features
    model = sm.OLS(y, sm.add_constant(X)).fit()

    # Loop through and remove features until none are significant at the given level
    while True:
        # Get the p-values of each feature
        p_values = model.pvalues.drop('const')

        # Find the highest p-value above the significance level
        max_p_value = p_values.max()
        if max_p_value > sig_level:
            # Remove the feature with the highest p-value
            remove_feature = p_values.idxmax()
            selected_cols.remove(remove_feature)

            # Refit the model with the remaining features
            model = sm.OLS(y, sm.add_constant(X[selected_cols])).fit()
        else:
            break

    # Print the final equation and variables
    print('Best model equation:')
    print('y = {:.6f}'.format(model.params[0]), end=' ')
    for i, col in enumerate(selected_cols):
        print('{:.6f} * {} +'.format(model.params[i+1], col), end=' ')
    print('')

    print('Variables in the best model:')
    print(selected_cols)

    return selected_cols


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

def step_wise_regress(X, y, sig_level=0.05):
    # Initialize the list of selected features
    selected_cols = list(X.columns)
    n_cols = len(selected_cols)

    # Loop through and remove features until none are significant at the given level or only one feature is left
    while len(selected_cols) > 1:
        # Fit a linear regression model with all selected features
        model = LinearRegression().fit(X[selected_cols], y)

        # Use recursive feature elimination to select the top 1 feature
        rfe = RFE(estimator=model, n_features_to_select=1)
        rfe.fit(X[selected_cols], y)

        # Get the support mask for the RFE
        support = rfe.support_

        if np.sum(support) > 0:
            # Get the indices of the selected features
            indices = np.where(support)[0]

            # Remove the feature with the lowest importance score
            remove_feature = selected_cols[indices[0]]
            selected_cols.remove(remove_feature)
        else:
            # If all features have importance score of 0, break the loop
            break

        # Check if the adjusted R-squared value is significant
        model = LinearRegression().fit(X[selected_cols], y)
        n = len(y)
        p = len(selected_cols)
        r_squared_adj = 1 - (1 - model.score(X[selected_cols], y)) * (n - 1) / (n - p - 1)

        if r_squared_adj < sig_level:
            # If adjusted R-squared is not significant, break the loop
            break

    # Print the final equation and variables
    print('Best model equation:')
    print('y = {:.6f}'.format(model.intercept_), end=' ')
    for i, col in enumerate(selected_cols):
        print('{:.6f} * {} +'.format(model.coef_[i], col), end=' ')
    print('')

    print('Variables in the best model:')
    print(selected_cols)

    return selected_cols


def find_best_model(df, result):

    cor_val = pearson_corr(df[var_list + ["MeanValue"]], "MeanValue")
    new_df = df[cor_val + ['MeanValue']]
    # new_df = df[var_list + ["MeanValue"]]

    # Split data into training and test
    X = new_df.iloc[:, :-1]
    # X = (X - X.mean()) / X.std()
    y = new_df['MeanValue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_test = (X_test - X_train.mean()) / X_train.std()
    X_train_norm = (X_train - X_train.mean()) / X_train.std()


    # Run stepwise regression on training set
    selected_cols = stepwise_selection(X_train_norm, y_train)
    model = sm.OLS(y_train, sm.add_constant(X_train[selected_cols])).fit()

    # Print equation and variables in the best model
    print('Best model equation:')
    print('y = {:.6f}'.format(model.params[0]), end=' ')
    result['intercept'] = model.params[0]
    for i, col in enumerate(selected_cols):
        result[col] = model.params[i+1]
        print('{:.6f} * {} +'.format(model.params[i+1], col), end=' ')
    print('')

    print('Variables in the best model:')
    print(selected_cols)

    X_test_selected = X_test[selected_cols]
    y_pred = model.predict(sm.add_constant(X_test_selected))
    r_squared = model.rsquared
    result['r2']=r_squared
    print('R-squared for the best model on test set:', r_squared)

    # Calculate RMSE for the best model on test set
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE for the best model on test set:', rmse)

    return result

def find_best_model2(df):
    # Calculate correlations between each feature and target variable
    cor_val = pearson_corr(df[var_list + ["MeanValue"]], 'MeanValue')

    # Create a new dataframe with columns containing the correlation values and target variable

    # new_df = df[cor_val + ['MeanValue']]

    # Split data into training and test sets and normalize the features
    X_train, X_test, y_train, y_test = train_test_split(new_df.iloc[:, :-1], new_df['MeanValue'], test_size=0.2,
                                                        random_state=42)
    scaler = StandardScaler()
    X_train_norm = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_norm = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

    # Run backward elimination to select the best features
    selected_cols = step_wise_regress(X_train_norm, y_train)

    # Fit the model on the selected features and get the coefficients
    model = sm.OLS(y_train, sm.add_constant(X_train_norm[selected_cols])).fit()
    coefs = pd.Series(model.params[1:], index=selected_cols)
    intercept = pd.Series(model.params[0], index=['intercept'])

    # Create a new dataframe to store the coefficients for all variables and intercepts
    coef_df = pd.concat([coefs, cor_val], axis=1, sort=False)
    coef_df.columns = ['coef', 'correlation']
    coef_df.loc[~coef_df.index.isin(selected_cols), 'coef'] = np.nan
    coef_df = coef_df.append(intercept)

    # Print the best model equation and variables
    print('Best model equation:')
    print('y = {:.6f}'.format(model.params[0]), end=' ')
    for i, col in enumerate(selected_cols):
        print('{:.6f} * {} +'.format(model.params[i + 1], col), end=' ')
    print('')
    print('Variables in the best model:')
    print(selected_cols)

    # Predict on the test set and calculate R-squared and RMSE
    X_test_selected = X_test_norm[selected_cols]
    y_pred = model.predict(sm.add_constant(X_test_selected))
    r_squared = model.rsquared
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('R-squared for the best model on test set:', r_squared)
    print('RMSE for the best model on test set:', rmse)

    r_squared = pd.Series(r_squared, index=['r_squared'])
    coef_df = coef_df.append(r_squared)

    return coef_df


def main():
    # ascfile here must have "totalPop" and "geoid" columns
    acsfile = workdir + '\\dexter_2309204431_extract.csv'
    uctresult = workdir + '\\tree_cover_tract_data.pkl'
    acs = pd.read_csv(acsfile, header=0, encoding="ISO-8859-1")
    utc = pd.read_pickle(uctresult)
    merged = pd.merge(acs, utc, left_on='geoid', right_on='NormID', how='inner')
    result_df = pd.DataFrame(index=[var_list + ["intercept"] + ["r2"]], columns=regions_dict.keys())
    for zone in regions_dict.keys():
        print("Processing zone: {}".format(zone))
        zone_df = by_zone(merged, zone)
        new_df = preprocess(zone_df)
        result_df[zone] = find_best_model(new_df, result_df[zone])

    result_df.to_csv("reg7.csv", index=True)
if __name__ == '__main__':
    main()