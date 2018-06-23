import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import calibration

import scipy.interpolate as interpolate

################################################################################
#                 Standard formulas for compound interest                      #
################################################################################

def interest_payments(amount, rate, term):
    """Total interest the borrower will pay if they pay on time monthly.
    See https://brownmath.com/bsci/loan.htm

    Parameters
    ----------
    amount : Amount loaned
    rate : interest rate fraction/year (decimal, not percent)
    term : Length of loan (years)

    Returns
    -------
    interest : Interest paid
    """
    # interest rate (fraction/month)
    i = rate/12.0
    # Number of payments
    N = term*12.0
    # Total payment per month
    P = i*amount / (1.0 - (1.0+i)**(-N))

    return P*N-amount


def calculate_interest_rate(amount_start, amount_end, term):
    """Calculate the interest rate for a loan paid monthly.

    Parameters
    ----------
    amount_start : Starting value of investment (loan amount)
    amount_end : End value of investment (total payed back including interest)
    term : length of loan in years

    Returns
    -------
    rate : interest rate in units of fraction/year
    """
    # Number of payments
    N = term*12.0

    P = amount_end/N
    y = P/amount_start

    ilist = np.logspace(-5, -1, 1000)
    ylist = np.array([i / (1.0 - (1+i)**(-N)) for i in ilist])
    iofy = interpolate.UnivariateSpline(ylist, ilist, k=1, s=0)
    i = iofy(y)
    return i*12


################################################################################
#                      Expectation value for payments                          #
################################################################################


def expectation_value_payment(loan_amount, rate, term, p_default, frac_recovered):
    """The expectation value of the amount the borrower pays back.

    Parameters
    ----------
    loan_amount : Amount loaned
    rate : interest rate fraction/year (decimal, not percent)
    term : Length of loan (years)
    p_default : probability of p_default
    frac_recovered : Estimate of the fraction recovered of the total owed
        (including interest) if the borrower defaults.

    Returns
    -------
    E[payment]
    """
    interest = interest_payments(loan_amount, rate, term)

    # Amount if borrower pays the loan in full
    total_paid_in_full = loan_amount + interest

    # Amount if borrower does not pay the loan in full
    total_not_paid_in_full = frac_recovered * total_paid_in_full

    return (1.0-p_default)*total_paid_in_full + p_default*total_not_paid_in_full


################################################################################
#                 Predict payments from a portfolio of loans                   #
################################################################################


def predict_default_probability(df, cal_clf, scaler):
    """Predict probability of default for all loans in the DataFrame.

    Parameters
    ----------
    df : DataFrame
    cal_clf : CalibratedClassifierCV object
    scaler : Scaler object

    Returns
    -------
    p_defaults : 1d-array
        Probability of default for each row in DataFrame
    """
    X = df.drop(['bad_loan', 'issue_d','total_rec_prncp', 'total_rec_int'], axis=1).values
    X = scaler.transform(X)
    p_defaults = cal_clf.predict_proba(X)[:, 1]

    return p_defaults


def total_payments_in_portfolio_predicted(
    df, cal_clf, scaler,
    frac_recovered=0.417, term=3.0):
    """Predicted amount the borrowers will pay back.

    Parameters
    ----------
    df : DataFrame
    cal_clf : CalibratedClassifierCV object
    scaler : Scaler object
        Feature scaling
    frac_recovered : float
        Amount you expect to recover on average if the borrowers default.
    term : float
        Length of loan in years

    Returns
    -------
    dict : dictionary of payment quantities
    """
    loan_amounts = df['loan_amnt'].values
    loan_rates = df['int_rate'].values / 100.0

    # Predict the probability of default
    p_defaults = predict_default_probability(df, cal_clf, scaler)

    # Predict the total payments you will receive
    tot_pay = 0.0
    nloans = len(loan_amounts)
    for i in range(nloans):
        loan_amount = loan_amounts[i]
        rate = loan_rates[i]
        p_default = p_defaults[i]
        tot_pay += expectation_value_payment(loan_amount, rate, term, p_default, frac_recovered)

    # Other quantities of interest
    tot_loan_amounts = np.sum(loan_amounts)
    tot_gains = tot_pay - tot_loan_amounts
    rate_eff = calculate_interest_rate(tot_loan_amounts, tot_pay, term)

    return {'tot_loan_amounts':tot_loan_amounts, 'tot_pay':tot_pay,
            'tot_gains':tot_gains, 'rate_eff':rate_eff}


def total_payments_in_portfolio_known(df, term=3.0):
    """The total amount that the borrowers did pay back.

    Parameters
    ----------
    df : DataFrame
    term : float
        Length of loan in years

    Returns
    -------
    dict : dictionary of payment quantities
    """
    loan_amounts = df['loan_amnt'].values
    total_principle_paid = df['total_rec_prncp'].values
    total_interest_paid = df['total_rec_int'].values

    tot_loan_amounts = np.sum(loan_amounts)
    tot_pay = np.sum(total_principle_paid) + np.sum(total_interest_paid)
    tot_gains = tot_pay - tot_loan_amounts
    rate_eff = calculate_interest_rate(tot_loan_amounts, tot_pay, term)

    return {'tot_loan_amounts':tot_loan_amounts, 'tot_pay':tot_pay,
            'tot_gains':tot_gains, 'rate_eff':rate_eff}


################################################################################
#                    Methods for optimally selecting loans                     #
################################################################################


def get_lowest_default_probability_loans(df, cal_clf, scaler, n):
    """Sort the loans by default probability then pick the n lowest.

    Parameters
    ----------
    df : DataFrame
    cal_clf : CalibratedClassifierCV object
    scaler : Scaler object
    n : int
        Number of loans.

    Returns
    -------
    p_default_chosen : 1d array
        Default probability of chosen loans
    df_chosen : DataFrame
        Data for the chosen loans
    """
    p_defaults = predict_default_probability(df, cal_clf, scaler)

    # Indices of n loans with lowest default probability
    p_sort_i = np.argsort(p_defaults)
    chosen_i = p_sort_i[:n]

    p_default_chosen = p_defaults[chosen_i]
    df_chosen = df.iloc[chosen_i, :]

    return p_default_chosen, df_chosen


def get_highest_predicted_roi_loans(df, cal_clf, scaler, n,
                                    frac_recovered=0.417, term=3.0):
    """Sort the loans by expected ROI in descending order, 
    then pick the n highest.

    Parameters
    ----------
    df : DataFrame
    cal_clf : CalibratedClassifierCV object
    scaler : Scaler object
    n : int
        Number of loans.

    Returns
    -------
    rate_chosen : 1d array
        Expectation value for ROI of chosen loans
    df_chosen : DataFrame
        Data for the chosen loans
    """
    loan_amounts = df['loan_amnt'].values
    loan_rates = df['int_rate'].values / 100.0

    # Predict the probability of default
    p_defaults = predict_default_probability(df, cal_clf, scaler)

    # Predict the effective ROI
    rate_eff = []
    nloans = len(loan_amounts)
    for i in range(nloans):
        loan_amount = loan_amounts[i]
        rate = loan_rates[i]
        p_default = p_defaults[i]
        e_pay = expectation_value_payment(loan_amount, rate, term, p_default, frac_recovered)
        r = calculate_interest_rate(loan_amount, e_pay, term)
        rate_eff.append(r)
    rate_eff = np.array(rate_eff)

    # Indices of n loans with highest expected ROI
    rate_sort_i = np.flip(np.argsort(rate_eff), axis=0)
    chosen_i = rate_sort_i[:n]

    rate_chosen = rate_eff[chosen_i]
    df_chosen = df.iloc[chosen_i, :]

    return rate_chosen, df_chosen
