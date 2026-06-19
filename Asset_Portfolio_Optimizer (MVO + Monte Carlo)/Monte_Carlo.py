# This script will perfrom a Monte Carlo simulation on the portfolios generated with the MVO_export script
# It assumes the average return and volatility are true
# annual rebalancing and dollar cost averaging are not yet simulated


# Notes to self:
# use functions to avoid clutter
# state function I/O and I/O datatypes
# use classes for data storage (Might be smart, but look into how to implement)
# set temporary variable anew each time they are used (within a scope) (for example, the sum needs to be reset before it is used again with +=)

import pandas as pd
import math
import numpy as np
from scipy.stats import rankdata

# import data (alrdy contains means and stdev for the individial assets)
cov_matrix_df = pd.read_csv("data/Covariance_MAtrix.csv")
cov_matrix = cov_matrix_df.values.tolist()

# portfolio structure + stdev & mean of original assets
# structure of portfolios_df has to be (in current version 5 rows, 11 columns):
'''portfolio1 portfolio2 portfolio3 ... mean_return_of_assets stdev_of_assets
   asset1weight asset1weight asset1weight ... meant_return_asset1 stdev_asset1
   2
   3
   4
   portfolio1stdev'''
portfolios_df = pd.read_csv("data/Example_Portfolios.csv")
stdev_list = [portfolios_df.iloc[i, portfolios_df.shape[1]-2] for i in range(portfolios_df.shape[0]-1)]
mean_list = [portfolios_df.iloc[i, portfolios_df.shape[1]-3] for i in range(portfolios_df.shape[0]-1)]
asset_names_list = [portfolios_df.iloc[i, portfolios_df.shape[1]-1] for i in range(portfolios_df.shape[0]-1)]

# this function generates the lower triangular matrix of the covariance matrix by cholesky decomposition
# Input: Covariance Matrix (2D list) (symmetric, nxn, postitive definite)
# Output: Lower triangular matrix (2D list)
# (indices -1 because in math we start at 1 but in programming at 0)
# (calculate once, save, reuse)
def cholesky_decompostion(cov_matrix):
    print("Cholesky function called")

    # initialize the lower triangular matrix of same size as n x n input matrix
    if len(cov_matrix) == len(cov_matrix[0]):
        LT_matrix = [[0 for a in range(len(cov_matrix[0]))] for b in range(len(cov_matrix))]
        n = len(cov_matrix)
        print(n)
    else:
        print("The imported covariance matrix is not of form n x n")

    # initialize the row and column identifiers
    i = 1
    j = 1

    # perform the cholesky algorithm (for diagonal and non diagonal case)
    # repeat until i == j == n (aka until the entire LT matrix is filled)
    iterator = 0 # this variable just tells us through how many iterations the while loop has gone for debugging purposes
    while i <= n and j <= n:

        print("While loop iteration: ", iterator)
        if i == j:
            print("case: i = j, i: ", i, ", j: ", j)
            # use formula for diagonal entries
            # calculate sum of L^2 terms first
            sum = 0
            k = 1
            while k <= j-1:
                sum += LT_matrix[j-1][k-1]**2
                k += 1

            LT_matrix[i-1][j-1] = math.sqrt(cov_matrix[i-1][j-1] - sum)

            i += 1
            j = 1
        elif i > j:
            print("case: i > j, i: ", i, ", j: ", j)
            # use formula for non - diagonal entries
            # calculate sum of L * L terms first (and reinitialize variable to prevent carryover errors)
            sum = 0
            k = 1
            while k <= j-1:
                sum += LT_matrix[i-1][k-1] * LT_matrix[j-1][k-1]
                k +=1

            LT_matrix[i-1][j-1] = (1/LT_matrix[j-1][j-1])*(cov_matrix[i-1][j-1] - sum)

            j += 1
        else:
            print("The element to be computed is not part of the lower triangular matrix")
        iterator += 1

    return LT_matrix

# this function generates n random years of returns for a given portfolio
# it uses the lower triangular matrix from the cholesky_decompostion function to correlate returns
# Input: portfolio dataframe, lower triangular matrix, index of column of target portfolio allocation
# output: returns a 2D list with rows = year and columns = return of a particular portfolio for that year
def annual_return(portfolios, LT_matrix, years_to_simulate):
    annual_returns = []

    for year in range(years_to_simulate):
        returns = [0 for i in range(portfolios.shape[0] - 1)]

        for i in range(len(returns)):
            # generate standard normal values with covariance matrix = identitiy matrix 
            returns[i] = np.random.randn()   

        # correlate the standard normal values by multiplication with cholesky factor from correlation matrix    
        correlated_returns = np.dot(returns, LT_matrix)

        # correct the correlated standard normal values to reflect the return distribution of the original assets
        for i in range(len(correlated_returns)):
            correlated_returns[i] = correlated_returns[i] * stdev_list[i] + mean_list[i]
        
        annual_returns.append(correlated_returns)

    return annual_returns

# caclulate annual returns for each portfolio and store in 2D list (j = which portfolio return, i = years)
# inputs: standardized dataframe containing portfolios, list of annual returns for all assets for any amount of years
# outputs: 2D list (each row corresponds to a year with each column representing the total return of 1 portfolio (from the dataframe) for that year)
def create_portfolio_returns_list(portfolios_df, annual_returns_list):

    portfolio_j_return = [[0 for a in range(portfolios_df.shape[1]-3)] for i in range(len(annual_returns_list))]
    for year in range(len(annual_returns_list)):
        for j in range(portfolios_df.shape[1]-3):
            for i in range(portfolios_df.shape[0]-1):
                portfolio_j_return[year][j] += portfolios_df.iloc[i, j] * annual_returns_list[year][i]
    
    return portfolio_j_return

# calculate final portfolio value for each portfolio afer (market conditions in the annual_returns_list)
# inputs: portfolio dataframe, integer representing starting capital, list of simulated annual returns per asset
# outputs: 1D list of final portfolio values after all simulated years (ordered like in portfolios_df)
def calculate_final_portfolio_values(portfolios_df, starting_capital, annual_returns_list, portfolio_returns):
    final_portfolio_value = []
    for portfolio in range(portfolios_df.shape[1] - 3):
        current_porftolio_value = starting_capital
        for year in range(len(annual_returns_list)):
            current_porftolio_value += current_porftolio_value*portfolio_returns[year][portfolio]
        final_portfolio_value.append(current_porftolio_value)
    
    return final_portfolio_value

# rank the portfolios according to their final value
# (if there are duplicate final values (from identical portfolio structures), they will all be in the ranking with portfolios that denote higher target returns in the dataframe having the higher index)
# (portfolio with higest value has the highest index in the order list)
# Inputs: A list of final portfolio values after x years of return
# Outputs: A ranked list of portfolio identifiers (e.g. [0,8,5,3] menas portfolio with index 3 has the highest return)
def rank_portfolios(final_portfolio_value):
    final_values_ranked = sorted(final_portfolio_value)
    seen = [] # to check for duplicate final values

    portfolio_ranks = ["none" for i in range(len(final_values_ranked))]
    for i, value in enumerate(final_values_ranked): # 'enumarate' operator returns a list of sets, with each set containing the value and its index of the original list
        if value not in seen:
            index = final_portfolio_value.index(value) # get the portfolio identifier of the final value in question 
            portfolio_ranks[i] = index # (0 = very left in original dataframe)
        elif value in seen: # if the value already occured, then there must be another portfolio with the same final value 1 to the right in the dataframe so + (seen.count(value)) to match its postition in the dataframe
            index = final_portfolio_value.index(value) + seen.count(value)
            portfolio_ranks[i] = index
            
        seen.append(final_portfolio_value[index])
    
    return portfolio_ranks


# this function simulates n simulations of m years of returns and outputs a ranking of portfolios according ot their average rank (index = portfolio indentifier, value = average rank)
# inputs: Datafarme of portfolios (structure see in import section), LT_matrix from cholesky decomposition, number of years per simulation, starting capital for each simulation, total number of simulations
def monte_carlo_simulation(portfolio_df, LT_matrix, years, starting_capital, number_of_simulations, original_asset_names):
    
    rank_sums = [0 for i in range(portfolios_df.shape[1]-3)] # sum of the ranks (according to final portfolio value) for each portfolio accross all simulations (index = index in portfolio dataframe, value = average rank)
    for i in range(number_of_simulations):
        annual_market_returns = annual_return(portfolio_df, LT_matrix, years)
        annual_portfolio_returns = create_portfolio_returns_list(portfolio_df,annual_market_returns)
        final_portfolio_values = calculate_final_portfolio_values(portfolio_df, starting_capital, annual_portfolio_returns, annual_portfolio_returns)
        portfolio_ranks = rank_portfolios(final_portfolio_values)

        for portfolio_index in portfolio_ranks:
            rank_sums[portfolio_index] += portfolio_ranks.index(portfolio_index)
    
    # get average ranks
    average_ranks = [0 for i in range(portfolios_df.shape[1]-3)]
    for i in range(len(rank_sums)):
        average_ranks[i]= rank_sums[i] / number_of_simulations

    for i in range(portfolio_df.shape[1]-3):
        print("Portfolio ",i+1, " (target return: ",i+1,"%)")
        print("Average rank: ", average_ranks[i])
        print("Asset allocation: ")
        for j in range(len(original_asset_names)):
            print(original_asset_names[j], ": ", portfolio_df.iloc[j, i])
        print("\n")




# turn covariance matrix into corrlation matrix (this allows adding our own stdev and mean to the simulated returns later)
correlation_matrix = [[0 for a in range(len(cov_matrix[0]))] for b in range(len(cov_matrix))]
for i in range(len(cov_matrix)):
    for j in range(len(cov_matrix[0])):
        correlation_matrix[i][j] = cov_matrix[i][j] / (stdev_list[i] * stdev_list[j])        

LT_matrix = cholesky_decompostion(correlation_matrix)
# these returns should now have the same mean, stdev, and correlation structure as the returns from example_returns.csv

# run the simulation 
monte_carlo_simulation(portfolios_df, LT_matrix, 50, 10000, 100, asset_names_list)

