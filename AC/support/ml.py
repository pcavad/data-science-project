'''
Machine Learning utility modul to support forecasting.
'''
# Python
import datetime
import os

# Thrid part
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme(style="darkgrid")

# Machine Learning
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import normaltest
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot, quarter_plot
from statsmodels.tsa.seasonal import seasonal_decompose

# support
from support import utils

def make_prediction(context, event):
    '''
    Makes a prediction to the end of the year using FB Prophet.

    Inputs:
        tseries: pd.Series >>> the timeseries with monthly total
        remove_outliers: bool = False >>> flatten the timeseries values above 75% percentile
        window_roll: int = 12 >>> number of months to calculate the rolling mean and std
        ts_seasonality: int = 6 >>> seasonality factor
        ts_model: str = 'multiplicative' >>> timeseries modeling ('multiplicative', 'additive')
        months_to_plot: int = 24 >>> months to plot for actual
        out_of_sample_months: int = 6 >>> number of months to predict
        figsize: tuple[int] = (16,6) >>> figure size
        barwidth: int = 10 >>> the width of the bars
        dump_path: Optional[str] >>> folder to save to Excel/csv
    Return:
        mean_squared_error(yhat)
    '''

    # Assign variables, assert type and set default where applicable
    try:
        tseries = context['tseries']
        assert isinstance(tseries, pd.Series), 'Timeseries is not a series'

        remove_outliers = event['remove_outliers']
        assert isinstance(remove_outliers, (bool, type(None)))\
            , 'remove_outliers must be bool or None'

        window_roll = event['window_roll']
        if window_roll is None:
            window_roll = 12
        else:
            assert isinstance(window_roll, int), 'window_roll must be int'
            assert 2 < window_roll < 13, 'window_roll must be between 3 and 12'

        ts_seasonality = event['ts_seasonality']
        if ts_seasonality is None:
            ts_seasonality = 6
        else:
            assert isinstance(ts_seasonality, int), 'seasonality must be int'
            assert 2 < ts_seasonality < 13, 'seasonality must be between 3 and 12'

        ts_model = event['ts_model']
        if ts_model is None:
            ts_model = 'multiplicative'
        else:
            assert ts_model in ['multiplicative', 'additive']\
                , 'Model must be additive or multiplicative'

        months_to_plot = event['months_to_plot']
        if months_to_plot is None:
            months_to_plot = 24
        else:
            assert isinstance(months_to_plot, int), 'months_to_plot must be int'
            assert 11 < months_to_plot < 25\
                , 'months_to_plot must be between 12 and 24'
            assert len(tseries) > months_to_plot,\
                'Timeseries must be longer than months to plot'

        out_of_sample_months = event['out_of_sample_months']
        if out_of_sample_months is None:
            out_of_sample_months = 24
        else:
            assert isinstance(out_of_sample_months, int)\
                , 'out_of_sample_months must be int'
            assert 2 < out_of_sample_months < 7\
                , 'out_of_sample_months must be between 3 and 6'
            assert months_to_plot > window_roll > out_of_sample_months\
                , 'inconsistent timelines'

        figsize = event['figsize']
        if figsize is None:
            figsize = (16,6)
        else:
            assert isinstance(figsize, tuple), 'figsize must be a tuple'
            assert len(figsize) == 2, 'figsize must be of length 2'
            assert isinstance(figsize[0], int)\
                , 'The first figsize dimension must be int'
            assert isinstance(figsize[1], int)\
                , 'The second figsize dimension must be int'

        barwidth = event['barwidth']
        if barwidth is None:
            barwidth = 10
        else:
            assert isinstance(barwidth, int), 'barwidth must be int'

        dump_path = event['dump_path']
        if dump_path:
            if not os.path.isdir(dump_path):
                raise Exception('dump_path is not a directory')

    except AssertionError as a:
        print(a)
        return None
    except TypeError as t:
        print(t)
        return None
    except Exception as e:
        print(e)
        return None

    ts = tseries.copy() # Prevents to modify the underlying data in the next step

    # Falatten the timeseries for values above 75% percentile
    if remove_outliers:
        max_75 = ts.describe()['75%']
        ts[ts > max_75] = max_75

    # Re-organize the time series for Prophet
    prophet_df = pd.DataFrame({'ds': ts.index.values, 'y': ts.values})

    # Create the object
    m = Prophet(weekly_seasonality=False, daily_seasonality=False)

    # Fit the model
    m.fit(prophet_df)

    # Create the time series for prediction using Prophet make_future_dataframe
    with_out_of_sample_dates_df =\
        m.make_future_dataframe(
            periods=out_of_sample_months,
            freq='M',
            include_history = False
            )

    # Predict
    forecast_prohpet_oos = m.predict(with_out_of_sample_dates_df)
    forecast_prohpet = m.predict(prophet_df)

    # Timeline to plot
    timeline = ts[-months_to_plot:] # From months_to_plot before

    # Plot timeseries with rolling mean and standard deviation
    fig1 = plt.figure()
    fig1.set_size_inches(figsize)
    plt.plot(timeline, 'b')
    plt.plot(timeline.rolling(window=window_roll).mean(), 'r') # rolling mean on window_roll months
    plt.plot(timeline.rolling(window=window_roll).std(), 'y') # std dev
    plt.title('Time series')
    plt.legend(
        ['Total by month', 'Rolling mean window = {}'.format(window_roll)
         , 'Rolling std dev window = {}'.format(window_roll)]
        )
    plt.show()

    print('Timeseries\n')
    print(normaltest(ts.values))
    print(f'mean: {np.mean(ts.values)}, std: {np.std(ts.values)}')
    print(adf_test(ts.values))

    # Plot Autocorrelation and partial autocorrelation for the entire timeseries
    fig2 = plt.figure()
    fig2.set_size_inches(figsize)
    ax1 = fig2.add_subplot(2,2,1)
    plot_acf(ts, ax=ax1) #, lags=nlags
    ax2 = fig2.add_subplot(2,2,2)
    plot_pacf(ts, ax=ax2) #, lags=nlags
    ax3 = fig2.add_subplot(2,2,3)
    # Plot months and quarterly plots for the entire timeseries
    month_plot(ts, ax=ax3)
    ax4 = fig2.add_subplot(2,2,4)
    quarter_plot(ts.resample('Q').sum(), ax=ax4)
    fig3 = plt.figure()
    fig3.set_size_inches(figsize)
    ax1 = fig3.add_subplot(1,2,1)
    # Plot histogram and boxplots for the entire timeseries
    plt.hist(ts, label='Timeseries', alpha=0.5)
    plt.hist(ts.diff().diff(), label='Timeseries diff by 2 lags', alpha=0.5)
    plt.legend()
    ax2 = fig3.add_subplot(1,2,2)
    sns.boxplot(x=ts.index.year, y=ts.values, ax=ax2)
    plt.grid(b=True)
    plt.show()

    # Plot timeseries decomposition
    ss_decomposition = seasonal_decompose(x=ts + 1e-4, model=ts_model, period=ts_seasonality)
    estimated_trend = ss_decomposition.trend
    estimated_seasonal = ss_decomposition.seasonal
    estimated_residual = ss_decomposition.resid

    fig4, axes = plt.subplots(3, 1, sharex=True, sharey=False)
    fig4.set_size_inches(figsize)

    axes[0].plot(estimated_trend, label='Trend')
    axes[0].legend(loc='upper left')

    axes[1].plot(estimated_seasonal, label='Seasonality')
    axes[1].legend(loc='upper left')

    axes[2].plot(estimated_residual, label='Residuals')
    axes[2].legend(loc='upper left')
    plt.show()

    # Plot forecast for the out of sample months
    fig = plt.figure(figsize = figsize)
    plt.bar( # Actual
        x = prophet_df['ds'][-out_of_sample_months:]
        , height = prophet_df['y'][-out_of_sample_months:]
        , color = 'b'
        , width = barwidth
        , label = 'Actual'
        )
    plt.bar( # Forecast
            x = with_out_of_sample_dates_df['ds']
            , height = forecast_prohpet_oos['yhat']
            , color = 'r'
            , alpha = 0.5
            , width = barwidth
            , label = 'Prediction'
            )
    plt.bar( # Forecast lowest
            x = with_out_of_sample_dates_df['ds']
            , height = forecast_prohpet_oos['yhat_lower']
            , color = 'g'
            , alpha = 0.5
            , width = barwidth
            , label = 'Prediction lowest'
            )
    plt.bar( # Forecast best
            x = with_out_of_sample_dates_df['ds']
            , height = forecast_prohpet_oos['yhat_upper']
            , color = 'y'
            , alpha = 0.5
            , width = barwidth
            , label = 'Prediction best'
            )
    plt.ylabel('Monthly revenue')
    plt.title('Forecast')
    plt.legend()
    ax = plt.gca()
    x_ticks = pd.concat(
        [prophet_df['ds'][-out_of_sample_months:],
         with_out_of_sample_dates_df['ds']]
        )
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks.astype('str'), rotation = 45)
    utils.add_value_labels(ax, symbol = '$',min_label=-50000) # add total amount on top of each bar
    plt.show()

    if dump_path:
        fig.savefig(os.path.join(dump_path, 'forecast.jpg'))

    date_start_forecast = with_out_of_sample_dates_df.min().item()
    date_end_forecast = with_out_of_sample_dates_df.max().item()
    str_date_current_year = str(datetime.datetime.today().year)

    print(f"Forecast between {str(date_start_forecast.date())} and {str(date_end_forecast.date())}")
    print('#########################################################')
    print('Normal: ', f'${forecast_prohpet_oos.yhat.sum():,.0f}')
    print('Lowest: ', f'${forecast_prohpet_oos.yhat_lower.sum():,.0f}')
    print('Best: ', f'${forecast_prohpet_oos.yhat_upper.sum():,.0f}')
    print('#########################################################')

    return mean_squared_error(prophet_df['y'], forecast_prohpet['yhat'])

def adf_test(t):
    '''
    Augmented Dickey-Fuller test.

    Input:
        t: pd.DataFrame >>> time series
    Return:
        Prints the test statistic, p-value, lags, critical values.
        If p-value < 0.01 then the series is stationary.
    '''

    dftest = adfuller(t)
    adf, pvalue = dftest[0], dftest[1]
    if pvalue < 0.01:
        print('reject null, the series is stationary: ', adf, pvalue)
    else:
        print('the series is non stationary:', adf, pvalue)
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic','p-value','Lags Used','Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
