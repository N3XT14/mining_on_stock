import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA

def twiggs_money_flow(data, period=21):
    """
    Computes the Twiggs Money Flow of a stock price chart.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the moving average.
    
    Returns:
    pd.DataFrame: A DataFrame containing the Twiggs Money Flow values.
    """
    # Compute typical price
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    
    # Compute the money flow raw values
    money_flow_raw = typical_price * data['Volume']
    
    # Compute the money flow volume
    money_flow_volume = money_flow_raw.copy()
    money_flow_volume[data['Close'] > data['Close'].shift(1)] = 0
    money_flow_volume = money_flow_volume.rolling(window=period).sum()
    
    # Compute the money flow volume difference
    money_flow_volume_diff = money_flow_volume.diff()
    
    # Compute the money flow ratio
    money_flow_ratio = money_flow_raw.rolling(window=period).sum() / money_flow_volume
    
    # Compute the Twiggs Money Flow
    twiggs_money_flow = money_flow_ratio * money_flow_volume_diff
    
    return pd.DataFrame({'Twiggs Money Flow': twiggs_money_flow})

def volume_underlay(data):
    """
    Computes the Volume Underlay of a stock price chart.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    
    Returns:
    pd.DataFrame: A DataFrame containing the Volume Underlay values.
    """
    volume_ma = data['Volume'].rolling(window=20).mean()
    volume_ratio = data['Volume'] / volume_ma
    volume_ratio_ma = volume_ratio.rolling(window=5).mean()
    volume_underlay = pd.DataFrame({'Price': data['Close'], 'Volume Ratio MA': volume_ratio_ma})
    return volume_underlay
def moving_average(data, period):
    """
    Computes the moving average of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the moving average.
    
    Returns:
    pd.Series: A Series containing the moving average values.
    """
    return data['Close'].rolling(window=period).mean()

def on_balance_volume(data):
    """
    Calculates the On Balance Volume (OBV) of a stock price.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    
    Returns:
    pd.Series: A Series containing the OBV values.
    """
    obv = pd.Series(data['Volume'], index=data.index)
    obv[0] = 0
    for i in range(1, len(data)):
        if data['Close'][i] > data['Close'][i-1]:
            obv[i] = obv[i-1] + data['Volume'][i]
        elif data['Close'][i] < data['Close'][i-1]:
            obv[i] = obv[i-1] - data['Volume'][i]
        else:
            obv[i] = obv[i-1]
    return obv
#Removed -  
def open_interest(data):
    """
    Calculates the Open Interest of a futures or options contract.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the futures or options data.
    
    Returns:
    pd.Series: A Series containing the Open Interest values.
    """
    return data['Open Interest']

def parabolic_sar(data, acceleration=0.02, maximum=0.2):
    """
    Calculates the Parabolic SAR of a stock price.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    acceleration (float): The acceleration factor for the indicator (default 0.02).
    maximum (float): The maximum value for the indicator (default 0.2).
    
    Returns:
    pd.Series: A Series containing the Parabolic SAR values.
    """
    high = data['High']
    low = data['Low']
    close = data['Close']
    af = acceleration
    max_af = maximum
    sar = close[0:len(close)]
    trend = True
    ep = low[0]
    for i in range(1, len(close)):
        if trend:
            sar[i] = sar[i-1] + af*(ep - sar[i-1])
            if high[i] > ep:
                ep = high[i]
                af = min(af + acceleration, max_af)
            if low[i] < sar[i]:
                trend = False
                sar[i] = ep
                ep = low[i]
                af = acceleration
        else:
            sar[i] = sar[i-1] + af*(ep - sar[i-1])
            if low[i] < ep:
                ep = low[i]
                af = min(af + acceleration, max_af)
            if high[i] > sar[i]:
                trend = True
                sar[i] = ep
                ep = high[i]
                af = acceleration
    return sar
def volume_underlay(data):
    """
    Computes the Volume Underlay of a stock price chart.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    
    Returns:
    pd.DataFrame: A DataFrame containing the Volume Underlay values.
    """
    volume_ma = data['Volume'].rolling(window=20).mean()
    volume_ratio = data['Volume'] / volume_ma
    volume_ratio_ma = volume_ratio.rolling(window=5).mean()
    volume_underlay = pd.DataFrame({'Price': data['Close'], 'Volume Ratio MA': volume_ratio_ma})
    return volume_underlay

def pivot_points(data):
    """
    Computes the pivot points for a given stock price data.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    
    Returns:
    dict: A dictionary containing the pivot point, support and resistance levels.
    """
    pivot = (data['High'] + data['Low'] + data['Close']) / 3
    r1 = (2 * pivot) - data['Low'].min()
    s1 = (2 * pivot) - data['High'].max()
    r2 = pivot + (data['High'].max() - data['Low'].min())
    s2 = pivot - (data['High'].max() - data['Low'].min())
    r3 = data['High'].max() + 2 * (pivot - data['Low'].min())
    s3 = data['Low'].min() - 2 * (data['High'].max() - pivot)
    return {'Pivot': pivot, 'Support 1': s1, 'Resistance 1': r1, 'Support 2': s2, 'Resistance 2': r2, 'Support 3': s3, 'Resistance 3': r3}

def positive_volume_index(data):
    """
    Computes the Positive Volume Index for a given stock price data.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    
    Returns:
    pd.Series: A Series containing the Positive Volume Index values.
    """
    pvi = pd.Series(data['Volume'].values)
    for i in range(1, len(data)):
        if data.iloc[i]['Volume'] > data.iloc[i-1]['Volume']:
            pvi[i] = pvi[i-1] + (data.iloc[i]['Close'] - data.iloc[i-1]['Close']) / data.iloc[i-1]['Close'] * pvi[i-1]
        else:
            pvi[i] = pvi[i-1]
    pvi.name = 'PVI'
    return pvi

def pretty_good_oscillator(data, period1=10, period2=20):
    """
    Computes the Pretty Good Oscillator (PGO) for a given stock price data.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period1 (int): The number of days to use for the first EMA calculation.
    period2 (int): The number of days to use for the second EMA calculation.
    
    Returns:
    pd.Series: A Series containing the PGO values.
    """
    ema1 = data['Close'].ewm(span=period1, min_periods=period1).mean()
    ema2 = data['Close'].ewm(span=period2, min_periods=period2).mean()
    pgo = ((ema1 - ema2) / data['Close'].rolling(window=period2).std()).rolling(window=period1).mean()
    return pgo

def price_momentum_oscillator(data, short_period=12, long_period=24):
    """
    Computes the Price Momentum Oscillator (PMO) of a stock price over a given short and long period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    short_period (int): The number of days to use for the short-term PMO calculation.
    long_period (int): The number of days to use for the long-term PMO calculation.
    
    Returns:
    pd.Series: A Series containing the PMO values.
    """
    roc_short = data['Close'].pct_change(periods=short_period)
    roc_long = data['Close'].pct_change(periods=long_period)
    pmo = 2 * roc_short - roc_long
    return pmo

def price_oscillator(data, short_period=12, long_period=26, signal_period=9):
    """
    Computes the Price Oscillator (PPO) of a stock price over a given short and long period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    short_period (int): The number of days to use for the short-term PPO calculation.
    long_period (int): The number of days to use for the long-term PPO calculation.
    signal_period (int): The number of days to use for the signal line calculation.
    
    Returns:
    pd.Series: A Series containing the PPO values.
    """
    short_ma = data['Close'].rolling(window=short_period).mean()
    long_ma = data['Close'].rolling(window=long_period).mean()
    ppo = 100 * (short_ma - long_ma) / long_ma
    signal_line = ppo.rolling(window=signal_period).mean()
    return ppo - signal_line


def price_rate_of_change(data, period):
    """
    Computes the Price Rate of Change (PROC) of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the PROC calculation.
    
    Returns:
    pd.Series: A Series containing the PROC values.
    """
    proc = 100 * (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)
    return proc

def price_volume_trend(data):
    """
    Computes the Price Volume Trend (PVT) of a stock price.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    
    Returns:
    pd.Series: A Series containing the PVT values.
    """
    pvt = ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1)) * data['Volume']
    pvt = pvt.cumsum()
    pvt.name = 'PVT'
    return pvt

import pandas as pd
import numpy as np

#Removed-  
def prime_number_bands(data, period=10, deviation=2):
    """
    Calculates the upper and lower prime number bands of a security's closing price.
    
    Parameters:
    -----------
    data : pd.DataFrame
        A DataFrame containing the security's data.
    period : int
        The number of periods to use in the calculation of the moving average.
    deviation : int
        The number of standard deviations to use in the calculation of the upper and lower bands.
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the upper and lower prime number bands.
    """
    # Calculate the simple moving average and standard deviation
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    
    # Calculate the upper and lower bands
    upper_band = sma + (deviation * std)
    lower_band = sma - (deviation * std)
    
    # Define a function to check if a number is prime
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                return False
        return True
    
    # Define a function to get the next prime number
    def get_next_prime(num):
        while not is_prime(num):
            num += 1
        return num
    
    # Apply the get_next_prime() function to the upper and lower bands
    upper_band = np.ceil(upper_band.apply(get_next_prime))
    lower_band = np.floor(lower_band.apply(get_next_prime))
    
    # Return a DataFrame with the upper and lower prime number bands
    return pd.DataFrame({'Upper Prime Number Band': upper_band, 'Lower Prime Number Band': lower_band})
#removed - 
def prime_number_oscillator(data, period=10):
    """
    Calculates the prime number oscillator of a security's closing price.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the security's data.
    period (int): The number of periods to use in the calculation.
    
    Returns:
    pd.Series: A Series containing the prime number oscillator values.
    """
    sma = data['Close'].rolling(window=period).mean()
    
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                return False
        return True
    
    def count_primes(num):
        count = 0
        for i in range(2, num+1):
            if is_prime(i):
                count += 1
        return count
    
    numerator = count_primes(data['Close']) - count_primes(sma)
    denominator = count_primes(data['Close'])
    pno = (numerator / denominator) * 100
    
    return pd.Series(pno, name='Prime Number Oscillator')

def KST(data, roc1, roc2, roc3, roc4, sma1, sma2, sma3, sma4):
    """
    Computes Pring's Know Sure Thing (KST) indicator for a stock price.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    roc1 (int): The first rate of change period.
    roc2 (int): The second rate of change period.
    roc3 (int): The third rate of change period.
    roc4 (int): The fourth rate of change period.
    sma1 (int): The first smoothed moving average period.
    sma2 (int): The second smoothed moving average period.
    sma3 (int): The third smoothed moving average period.
    sma4 (int): The fourth smoothed moving average period.
    
    Returns:
    pd.Series: A Series containing the KST values.
    """
    rocma1 = data['Close'].pct_change(roc1).rolling(sma1).mean()
    rocma2 = data['Close'].pct_change(roc2).rolling(sma2).mean()
    rocma3 = data['Close'].pct_change(roc3).rolling(sma3).mean()
    rocma4 = data['Close'].pct_change(roc4).rolling(sma4).mean()
    kst = rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4
    signal = kst.rolling(window=9).mean()
    return kst - signal

def SpecialK(data, roc1, roc2, roc3, roc4, sma1, sma2, sma3, sma4):
    """
    Computes Pring's Special K indicator for a stock price.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    roc1 (int): The first rate of change period.
    roc2 (int): The second rate of change period.
    roc3 (int): The third rate of change period.
    roc4 (int): The fourth rate of change period.
    sma1 (int): The first smoothed moving average period.
    sma2 (int): The second smoothed moving average period.
    sma3 (int): The third smoothed moving average period.
    sma4 (int): The fourth smoothed moving average period.
    
    Returns:
    pd.Series: A Series containing the Special K values.
    """
    rocma1 = data['Close'].pct_change(roc1).rolling(sma1).mean()
    rocma2 = data['Close'].pct_change(roc2).rolling(sma2).mean()
    rocma3 = data['Close'].pct_change(roc3).rolling(sma3).mean()
    rocma4 = data['Close'].pct_change(roc4).rolling(sma4).mean()
    kst = rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4
    signal = kst.rolling(window=9).mean()
    ema1 = kst.ewm(span=10, adjust=False).mean()
    ema2 = signal.ewm(span=15, adjust=False).mean()
    return ema1 - ema2

def projected_aggregate_volume(data, period):
    """
    Computes the Projected Aggregate Volume (PAV) of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the PAV calculation.
    
    Returns:
    pd.Series: A Series containing the PAV values.
    """
    volume_sum = data['Volume'].rolling(window=period).sum()
    pav = (volume_sum / period) * data['Close']
    pav.name = 'PAV'
    return pav

def projected_volume_at_time(data, period):
    """
    Computes the Projected Volume at Time (PVT) of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the PVT calculation.
    
    Returns:
    pd.Series: A Series containing the PVT values.
    """
    close_diff = data['Close'].diff()
    volume_diff = data['Volume'].diff()
    pvt = ((close_diff / data['Close'].shift(periods=1)) * volume_diff) + data['Volume'].shift(periods=1)
    pvt.name = 'PVT'
    return pvt

def psychological_line(data, period):
    """
    Computes the Psychological Line (PL) of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the PL calculation.
    
    Returns:
    pd.Series: A Series containing the PL values.
    """
    close_diff = data['Close'].diff()
    up_count = close_diff.where(close_diff > 0, 0).rolling(window=period).count()
    down_count = close_diff.where(close_diff < 0, 0).rolling(window=period).count()
    return up_count / (up_count + down_count)

def qstick(data, period):
    """
    Computes the QStick of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the QStick calculation.
    
    Returns:
    pd.Series: A Series containing the QStick values.
    """
    qs = (data['Close'] - data['Open'].rolling(window=period).mean())
    qs.name = 'QS'
    return qs

def ravi(data, short_period, long_period):
    """
    Computes the Range Action Verification Index (RAVI) of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    short_period (int): The number of days to use for the short EMA calculation.
    long_period (int): The number of days to use for the long EMA calculation.
    
    Returns:
    pd.Series: A Series containing the RAVI values.
    """
    short_ema = data['Close'].ewm(span=short_period).mean()
    long_ema = data['Close'].ewm(span=long_period).mean()
    ravi = ((short_ema - long_ema) / long_ema) * 100
    return ravi

def rsi(data, period):
    """
    Computes the Relative Strength Index (RSI) of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the RSI calculation.
    
    Returns:
    pd.Series: A Series containing the RSI values.
    """
    delta = data['Close'].diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def rsi_divergence(data, rsi_period, signal_period):
    """
    Computes the RSI Divergence of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    rsi_period (int): The number of days to use for the RSI calculation.
    signal_period (int): The number of days to use for the signal calculation.
    
    Returns:
    pd.Series: A Series containing the RSI Divergence values.
    """
    rsi_value = rsi(data, rsi_period)
    rsi_signal = rsi_value.rolling(window=signal_period).mean()
    rsi_div = rsi_value - rsi_signal
    return rsi_div

def rainbow_moving_average(data, n, k=1.0):
    """
    Computes the Rainbow Moving Average of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    n (int): The number of days to use for the moving average.
    k (float): A smoothing factor.
    
    Returns:
    pd.Series: A Series containing the Rainbow Moving Average values.
    """
    weights = np.arange(1, n+1)
    weights = np.power(weights, k)
    weights_sum = np.sum(weights)
    weights = weights / weights_sum
    rma = np.convolve(data['Close'], weights, mode='valid')
    rma = pd.Series(rma, index=data.index[n-1:])
    return rma

def rainbow_oscillator(data, n, k=1.0):
    """
    Computes the Rainbow Oscillator of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    n (int): The number of days to use for the Rainbow Moving Average calculation.
    k (float): A smoothing factor for the Rainbow Moving Average.
    
    Returns:
    pd.Series: A Series containing the Rainbow Oscillator values.
    """
    rma_short = rainbow_moving_average(data, n, k)
    rma_long = rainbow_moving_average(data, 2*n, k)
    rainbow_osc = rma_short - rma_long
    return rainbow_osc

def random_walk_index(data, period):
    """
    Computes the Random Walk Index of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the Random Walk Index calculation.
    
    Returns:
    pd.Series: A Series containing the Random Walk Index values.
    """
    delta = data['Close'].diff()
    up_down = np.where(delta >= 0, 1, -1)
    rwi = up_down * delta.abs().rolling(window=period).sum().apply(np.sqrt)
    return rwi

def relative_vigor_index(data, period):
    """
    Computes the Relative Vigor Index of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the Relative Vigor Index calculation.
    
    Returns:
    pd.Series: A Series containing the Relative Vigor Index values.
    """
    typical_price = (data['High'] + data['Low'] + 2*data['Close']) / 4
    rvi = typical_price.diff(period) / typical_price.rolling(window=period).sum().diff(period)
    return rvi

def schaff_trend_cycle(data, period):
    """
    Computes the Schaff Trend Cycle of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the Schaff Trend Cycle calculation.
    
    Returns:
    pd.Series: A Series containing the Schaff Trend Cycle values.
    """
    ema1 = data['Close'].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    macd = ema1 - ema2
    stc = macd.ewm(span=period, adjust=False).mean()
    return stc

def standard_deviation(data, period):
    """
    Computes the Standard Deviation of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the Standard Deviation calculation.
    
    Returns:
    pd.Series: A Series containing the Standard Deviation values.
    """
    return data['Close'].rolling(window=period).std()

def stochastic_divergence(data, period):
    """
    Computes the Stochastic Divergence of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the Stochastic Divergence calculation.
    
    Returns:
    pd.Series: A Series containing the Stochastic Divergence values.
    """
    stochastic = (data['Close'] - data['Low'].rolling(window=period).min()) / (data['High'].rolling(window=period).max() - data['Low'].rolling(window=period).min())
    divergence = stochastic - stochastic.rolling(window=period).mean()
    return divergence

def stochastic_momentum_index(data, period=14, smooth=3):
    """
    Calculates the Stochastic Momentum Index (SMI) of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the SMI calculation.
    smooth (int): The number of days to use for smoothing the SMI.
    
    Returns:
    pd.Series: A Series containing the SMI values.
    """
    highest_high = data['High'].rolling(window=period).max()
    lowest_low = data['Low'].rolling(window=period).min()
    k = 100 * (data['Close'] - lowest_low) / (highest_high - lowest_low)
    sma_k = k.rolling(window=smooth).mean()
    sma_sma_k = sma_k.rolling(window=smooth).mean()
    smi = k - sma_sma_k
    return smi

def stochastic_rsi(data, period=14, k=3, d=3):
    """
    Calculates the Stochastic RSI (StochRSI) of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the StochRSI calculation.
    k (int): The number of days to use for smoothing the StochRSI.
    d (int): The number of days to use for smoothing the StochRSI signal.
    
    Returns:
    pd.Series: A Series containing the StochRSI values.
    """
    rsi_ = rsi(data, period)
    highest_rsi = rsi_.rolling(window=k).max()
    lowest_rsi = rsi_.rolling(window=k).min()
    stochrsi = 100 * (rsi_ - lowest_rsi) / (highest_rsi - lowest_rsi)
    sma_stochrsi = stochrsi.rolling(window=d).mean()
    return sma_stochrsi

def stochastics(data, period=14, k=3, d=3):
    """
    Calculates the Stochastic Oscillator of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the Stochastic Oscillator calculation.
    k (int): The number of days to use for smoothing the Stochastic Oscillator.
    d (int): The number of days to use for smoothing the Stochastic Oscillator signal.
    
    Returns:
    pd.Series: A Series containing the Stochastic Oscillator values.
    """
    highest_high = data['High'].rolling(window=period).max()
    lowest_low = data['Low'].rolling(window=period).min()
    k = 100 * (data['Close'] - lowest_low) / (highest_high - lowest_low)
    sma_k = k.rolling(window=period).mean()
    sma_d = sma_k.rolling(window=period).mean()
    return sma_d

def supertrend(data, period=7, multiplier=3):
    """
    Computes the Supertrend indicator for a given stock price data.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of periods to use for the Supertrend calculation.
    multiplier (int): The multiplier value for the Supertrend calculation.
    
    Returns:
    pd.DataFrame: A DataFrame containing the Supertrend values.
    """
    high = data['High']
    low = data['Low']
    close = data['Close']
    atr = pd.Series((high.combine(low, min).combine(close.shift(), max) - low.combine(close.shift(), max)).rolling(period).mean())
    upper_band = pd.Series((high + low) / 2 + multiplier * atr)
    lower_band = pd.Series((high + low) / 2 - multiplier * atr)
    uptrend = pd.Series(data['Close'] > lower_band)
    downtrend = pd.Series(data['Close'] < upper_band)
    trend = pd.Series(0, index=data.index)
    trend[0] = 0
    for i in range(1, len(data)):
        if uptrend.iloc[i]:
            trend.iloc[i] = max(trend.iloc[i-1], 0)
        elif downtrend.iloc[i]:
            trend.iloc[i] = min(trend.iloc[i-1], 0)
        else:
            trend.iloc[i] = trend.iloc[i-1]
    supertrend = pd.Series(0, index=data.index)
    supertrend[0] = 0
    for i in range(1, len(data)):
        if trend.iloc[i-1] < 0 and trend.iloc[i] > 0:
            supertrend.iloc[i] = lower_band.iloc[i]
        elif trend.iloc[i-1] > 0 and trend.iloc[i] < 0:
            supertrend.iloc[i] = upper_band.iloc[i]
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1] if trend.iloc[i] > 0 else supertrend.iloc[i-1]
    return supertrend

def trix(data, period=15):
    """
    Computes the Triple Exponential Moving Average (TRIX) indicator for a given stock price data.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of periods to use for the TRIX calculation.
    
    Returns:
    pd.DataFrame: A DataFrame containing the TRIX values.
    """
    close = data['Close']
    ema1 = close.ewm(span=period, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, min_periods=period).mean()
    trix = pd.Series(ema3.pct_change(periods=1), name='TRIX')
    return trix

from statsmodels.tsa.arima.model import ARIMA
#Removed
def time_series_forecast(data, order=(1, 1, 1)):
    """
    Computes the Time Series Forecast of a stock price using the ARIMA model.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    order (tuple): The (p, d, q) order of the ARIMA model to use.
    
    Returns:
    pd.Series: A Series containing the Time Series Forecast values.
    """
    model = ARIMA(data['Close'], order=order)
    model_fit = model.fit()
    return model_fit.predict(start=1, end=len(data), dynamic=False)


def trade_volume_index(data):
    """
    Computes the Trade Volume Index of a stock price.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    
    Returns:
    pd.Series: A Series containing the Trade Volume Index values.
    """
    tvi = pd.Series(0.0, index=data.index)
    for i in range(1, len(data)):
        if data['Close'][i] > data['Close'][i - 1]:
            tvi[i] = tvi[i - 1] + (data['Volume'][i] * (data['Close'][i] - data['Close'][i - 1]) / data['Close'][i])
        elif data['Close'][i] < data['Close'][i - 1]:
            tvi[i] = tvi[i - 1] - (data['Volume'][i] * (data['Close'][i - 1] - data['Close'][i]) / data['Close'][i - 1])
        else:
            tvi[i] = tvi[i - 1]
    return tvi

def trend_intensity_index(data, period):
    """
    Computes the Trend Intensity Index of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the Trend Intensity Index calculation.
    
    Returns:
    pd.Series: A Series containing the Trend Intensity Index values.
    """
    tr = pd.DataFrame({
        'h-l': data['High'] - data['Low'],
        'h-pc': abs(data['High'] - data['Close'].shift(1)),
        'l-pc': abs(data['Low'] - data['Close'].shift(1))
    })
    tr_sum = tr.sum(axis=1)
    atr = tr_sum.rolling(window=period).mean()
    ti = pd.Series((data['Close'] - data['Close'].shift(period)) / atr, name='trend_intensity_index')
    return ti

def true_range(data):
    """
    Computes the True Range of a stock price.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    
    Returns:
    pd.Series: A Series containing the True Range values.
    """
    tr = pd.DataFrame({
        'h-l': data['High'] - data['Low'],
        'h-pc': abs(data['High'] - data['Close'].shift(1)),
        'l-pc': abs(data['Low'] - data['Close'].shift(1))
    })
    tr_max = tr.max(axis=1)
    return tr_max

def typical_price(data):
    """
    Computes the Typical Price of a stock price.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    
    Returns:
    pd.Series: A Series containing the Typical Price values.
    """
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    return tp
# Removed -
def ulcer_index(data, period):
    """
    Computes the Ulcer Index of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the Ulcer Index calculation.
    
    Returns:
    pd.Series: A Series containing the Ulcer Index values.
    """
    tp = typical_price(data)
    ui = ((tp.rolling(window=period).apply(lambda x: (x - x.max()) ** 2)).rolling(window=period).mean()) ** 0.5
    return ui

def ultimate_oscillator(data, period1=7, period2=14, period3=28, weight1=4, weight2=2, weight3=1):
    """
    Computes the Ultimate Oscillator of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period1 (int): The number of days to use for the first period.
    period2 (int): The number of days to use for the second period.
    period3 (int): The number of days to use for the third period.
    weight1 (int): The weight to use for the first period.
    weight2 (int): The weight to use for the second period.
    weight3 (int): The weight to use for the third period.
    
    Returns:
    pd.Series: A Series containing the Ultimate Oscillator values.
    """
    min_low_or_prior_close = pd.concat([data['Close'].shift(1), data['Low']], axis=1).min(axis=1)
    max_high_or_prior_close = pd.concat([data['Close'].shift(1), data['High']], axis=1).max(axis=1)
    
    bp = data['Close'] - min_low_or_prior_close
    tr = max_high_or_prior_close - min_low_or_prior_close
    
    avg1 = bp.rolling(window=period1).sum() / tr.rolling(window=period1).sum()
    avg2 = bp.rolling(window=period2).sum() / tr.rolling(window=period2).sum()
    avg3 = bp.rolling(window=period3).sum() / tr.rolling(window=period3).sum()
    
    ult_osc = ((weight1 * avg1) + (weight2 * avg2) + (weight3 * avg3)) / (weight1 + weight2 + weight3)
    
    return ult_osc

def vwap(data):
    """
    Computes the Volume-Weighted Average Price (VWAP) of a stock over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    
    Returns:
    pd.Series: A Series containing the VWAP values.
    """
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (data['Volume'] * tp).cumsum() / data['Volume'].cumsum()
    
    return vwap
#Removed - 
def valuation_lines(data, earnings_col='Earnings', cash_flow_col='Cash Flow', period=10):
    """
    Computes the Valuation Lines of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    earnings_col (str): The name of the column containing the earnings data.
    cash_flow_col (str): The name of the column containing the cash flow data.
    period (int): The number of days to use for the moving average.
    
    Returns:
    pd.DataFrame: A DataFrame containing the Valuation Lines values.
    """
    earnings_ma = data[earnings_col].rolling(window=period).mean()
    cash_flow_ma = data[cash_flow_col].rolling(window=period).mean()
    valuation_lines = pd.DataFrame({'Price': data['Close'], 'Earnings MA': earnings_ma, 'Cash Flow MA': cash_flow_ma})
    return valuation_lines

def vertical_horizontal_filter(data, period=28):
    """
    Computes the Vertical Horizontal Filter of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the VHF calculation.
    
    Returns:
    pd.Series: A Series containing the VHF values.
    """
    price_range = data['High'] - data['Low']
    abs_diff = abs(price_range.diff(periods=1))
    total_price_range = price_range.rolling(window=period).sum()
    vhf = abs_diff.rolling(window=period).sum() / total_price_range
    return vhf

def volatility_clone(data, period):
    """
    Computes the Volatility Clone of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the Volatility Clone calculation.
    
    Returns:
    pd.Series: A Series containing the Volatility Clone values.
    """
    close = data['Close']
    diff = close.diff()
    atr = pd.DataFrame(abs(data['High'] - data['Low']))
    atr = atr.rolling(window=period).mean()
    std = pd.DataFrame(diff.rolling(window=period).std())
    vc = std / atr
    return vc

def volume_oscillator(data, period1, period2):
    """
    Computes the Volume Oscillator of a stock price over two given periods of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period1 (int): The number of days to use for the short-term period.
    period2 (int): The number of days to use for the long-term period.
    
    Returns:
    pd.Series: A Series containing the Volume Oscillator values.
    """
    volume = data['Volume']
    short_term = volume.rolling(window=period1).mean()
    long_term = volume.rolling(window=period2).mean()
    vo = (short_term - long_term) / long_term
    return vo

def volume_profile(data, period):
    """
    Computes the Volume Profile of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the Volume Profile calculation.
    
    Returns:
    pd.DataFrame: A DataFrame containing the Volume Profile values.
    """
    prices = pd.DataFrame(data[['Open', 'High', 'Low', 'Close']].mean(axis=1), columns=['Price'])
    prices['Volume'] = data['Volume']
    prices['Time'] = range(len(prices))
    prices['Period'] = prices['Time'] // period
    vp = prices.groupby('Period').apply(lambda x: x.set_index('Time')['Volume'].reindex(range(period)).fillna(0))
    return vp

def volume_rate_of_change(data, period):
    """
    Computes the Volume Rate of Change of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the Volume Rate of Change calculation.
    
    Returns:
    pd.Series: A Series containing the Volume Rate of Change values.
    """
    volume = data['Volume']
    roc = volume.pct_change(periods=period)
    return roc

    """
    Computes the Volume Underlay of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the Volume Underlay calculation.
    
    Returns:
    pd.Series: A Series containing the Volume Underlay values.
    """
    vol = data['Volume']
    vol_ma = vol.rolling(window=period).mean()
    vol_underlay = vol / vol_ma
    return vol_underlay

def vortex_indicator(dataframe, window = 14):
    df = dataframe.copy()
    try:
        df['tr1'] = (df['High']- df['Low'])
        df['tr2'] = (df['High']- df['Close'].shift(1))
        df['tr3'] = (df['Low'] - df['Close'].shift(1))
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].values.max(1)
    except ValueError:
        return np.nan
    min_periods = window
    df['trn'] = df['true_range'].rolling(window, min_periods=min_periods).sum()
    df['vmp'] = np.abs(df['High'] - df['Low'].shift(1))
    df['vmm'] = np.abs(df['Low'] - df['High'].shift(1))
    vip = df['vmp'].rolling(window, min_periods=min_periods).sum() / df['trn']
    vin = df['vmm'].rolling(window, min_periods=min_periods).sum() / df['trn']
    return vip-vin

def williams_r(data, period):
    """
    Computes the Williams %R indicator of a stock price over a given period of time.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    period (int): The number of days to use for the Williams %R calculation.
    
    Returns:
    pd.Series: A Series containing the Williams %R values.
    """
    highest_high = data['High'].rolling(window=period).max()
    lowest_low = data['Low'].rolling(window=period).min()
    williams_r = (highest_high - data['Close']) / (highest_high - lowest_low) * -100
    return williams_r

def zigzag(data, deviation):
    """
    Computes the ZigZag financial indicator of a stock price over a given deviation threshold.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing the stock price data.
    deviation (float): The minimum price change percentage required to change the trend.
    
    Returns:
    pd.Series: A Series containing the ZigZag values.
    """
    # Compute the differences between consecutive closing prices
    diffs = data['Close'].diff()
    
    # Compute the absolute percentage changes between consecutive closing prices
    changes = np.abs(diffs / data['Close'].shift(1)) * 100
    
    # Initialize the trend and pivot variables
    trend = 0
    pivot = np.nan
    
    # Initialize the ZigZag values with the first closing price
    zigzag = pd.Series(index=data.index, data=data['Close'].values)
    
    # Iterate over the changes and update the trend and pivot as needed
    for i, change in enumerate(changes):
        # If the trend is up and the change is negative or the trend is down and the change is positive,
        # update the trend and pivot
        if (trend == 1 and change < -deviation) or (trend == -1 and change > deviation):
            trend = -trend
            pivot = data['Close'].iloc[i]
            
        # If the trend is up and the change is positive or the trend is down and the change is negative,
        # update the ZigZag value with the last pivot value
        elif (trend == 1 and change > deviation) or (trend == -1 and change < -deviation):
            zigzag.iloc[i] = pivot
            
        # If the trend is unchanged, update the ZigZag value with the last ZigZag value
        else:
            zigzag.iloc[i] = zigzag.iloc[i-1]
    
    return zigzag
    
import ta
import ta.volume
import sklearn
import pandas as pd
import numpy as np
from ta.trend import MACD
from sklearn.linear_model import LinearRegression

def ease_of_movement(df):
    em = ta.volume.EaseOfMovementIndicator(high=df["High"], low=df["Low"], volume=df["Volume"])
    df["EOM"] = em.ease_of_movement()
    return df['EOM']
#1
def calculate_adx(df, window=14):
    """Calculate Average Directional Index (ADX)"""
    # Calculate true range (TR)
    df = df.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    # Calculate directional movement (+DM, -DM)
    df['P-DM'] = df['High'] - df['High'].shift(1)
    df['N-DM'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['P-DM'] > df['N-DM']) & (df['P-DM'] > 0), df['P-DM'], 0)
    df['-DM'] = np.where((df['N-DM'] > df['P-DM']) & (df['N-DM'] > 0), df['N-DM'], 0)
    
    # Smoothed directional movement (+SDM, -SDM)
    df['+SDM'] = df['+DM'].rolling(window=window).mean()
    df['-SDM'] = df['-DM'].rolling(window=window).mean()
    
    # Calculate positive directional index (+DI) and negative directional index (-DI)
    df['+DI'] = (df['+SDM'] / df['TR'].rolling(window=window).sum()) * 100
    df['-DI'] = (df['-SDM'] / df['TR'].rolling(window=window).sum()) * 100
    
    # Calculate average directional index (ADX)
    df['DX'] = (np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].rolling(window=window).mean()
    
    # Drop intermediate columns used in calculation
    df.drop(['H-L', 'H-PC', 'L-PC', 'P-DM', 'N-DM', '+DM', '-DM', '+SDM', '-SDM', '+DI', '-DI', 'DX'], axis=1, inplace=True)
    return df['ADX']


#2
def atr_bands(df, n=20, m=2):
    """
    Calculates ATR Bands for a given data frame.

    Parameters:
        -- df (pd.DataFrame): Data frame containing OHLC (Open, High, Low, Close) data.
        -- n (int): Number of periods to calculate ATR. Default is 20.
        -- m (float): Multiplier for ATR bands. Default is 2.

    Returns:
        -- pd.DataFrame: Data frame with ATR Bands columns.
    """
    # Calculate True Range (TR)
    df['TR'] = np.max([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)),
                      np.abs(df['Low'] - df['Close'].shift(1))], axis=0)

    # Calculate Average True Range (ATR)
    df['ATR1'] = df['TR'].rolling(n).mean()

    # Calculate ATR Bands
    df['Upper Band'] = df['Close'] + (m * df['ATR1'])
    df['Lower Band'] = df['Close'] - (m * df['ATR1'])

    return df[['Upper Band', 'Lower Band']]
'''
def atr_trailing_stops(df, n=20, m=3):
    """
    Calculates ATR Trailing Stops for a given data frame.

    Parameters:
        -- df (pd.DataFrame): Data frame containing OHLC (Open, High, Low, Close) data.
        -- n (int): Number of periods to calculate ATR. Default is 20.
        -- m (float): Multiplier for ATR. Default is 3.

    Returns:
        -- pd.DataFrame: Data frame with ATR Trailing Stops columns.
    """
    # Calculate True Range (TR)
    df['TR'] = np.max([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)),
                      np.abs(df['Low'] - df['Close'].shift(1))], axis=0)

    # Calculate Average True Range (ATR)
    df['ATR'] = df['TR'].rolling(n).mean()

    # Calculate ATR Trailing Stops
    df['ATR Trailing Stop Long'] = df['Low'] + (m * df['ATR'])
    df['ATR Trailing Stop Short'] = df['High'] - (m * df['ATR'])

    return df
'''
#3
def accumulation_distribution(df):
    ad = ta.volume.AccDistIndexIndicator(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"])
    df["ADI"] = ad.acc_dist_index()
    return df['ADI']
#4
def accumulative_swing_index(df):
    """
    Calculates Accumulative Swing Index (ASI) for a given data frame.

    Parameters:
        -- df (pd.DataFrame): Data frame containing OHLCV (Open, High, Low, Close, Volume) data.

    Returns:
        -- pd.DataFrame: A DataFrame with ASI values as a new column added to the input data frame.
    """
    # Calculate necessary intermediate values
    hl = df['High'] - df['Low']
    hc1 = abs(df['High'] - df['Close'].shift(1))
    lc1 = abs(df['Low'] - df['Close'].shift(1))

    # Calculate the True Range (TR) and the Directional Movement (DM)
    tr = hl.where(hl >= hc1, hc1)
    tr = tr.where(tr >= lc1, lc1)
    dm_plus = (df['High'] - df['High'].shift(1)).where(df['High'] > df['High'].shift(1), 0)
    dm_minus = (df['Low'].shift(1) - df['Low']).where(df['Low'] < df['Low'].shift(1), 0)

    # Calculate the Directional Movement Index (DX)
    tr_ma = tr.rolling(window=14).mean()
    dm_plus_ma = dm_plus.rolling(window=14).mean()
    dm_minus_ma = dm_minus.rolling(window=14).mean()
    dx = 100 * (abs(dm_plus_ma - dm_minus_ma) / tr_ma)

    # Calculate the Accumulative Swing Index (ASI)
    asi = df['Close'].shift(1) + (0.5 * tr) + (0.25 * df['Close'].shift(1)) - (0.25 * df['Open'].shift(1))
    asi = asi + (0.125 * df['Close']) - (0.125 * df['Open'])
    asi = asi + (0.25 * df['Close'].shift(1)) - (0.25 * df['Close'].shift(2))
    asi = asi + (0.0625 * df['Close'].shift(2)) - (0.0625 * df['Open'].shift(2))
    asi = asi + (0.03125 * df['Close'].shift(3)) - (0.03125 * df['Open'].shift(3))
    asi = asi * (1 + dx) + asi.shift(1)

    # Add ASI values as a new column to the input data frame
    df['ASI'] = asi

    return df['ASI']


#5
def alligator(df, jaw_length=13, teeth_length=8, lips_length=5):
    """
    Calculates the Alligator indicator for a given data frame.

    Parameters:
        -- df (pd.DataFrame): Data frame containing OHLC (Open, High, Low, Close) data.
        -- jaw_length (int): Period length for the "Jaw" (blue) line. Default is 13.
        -- teeth_length (int): Period length for the "Teeth" (red) line. Default is 8.
        -- lips_length (int): Period length for the "Lips" (green) line. Default is 5.

    Returns:
        -- pd.DataFrame: Data frame containing Alligator indicator values (Jaw, Teeth, Lips).
    """
    # Calculate Moving Averages
    df['Jaw'] = df['Close'].rolling(window=jaw_length).mean().shift(1)
    df['Teeth'] = df['Close'].rolling(window=teeth_length).mean().shift(1)
    df['Lips'] = df['Close'].rolling(window=lips_length).mean().shift(1)

    return df.loc[:, ['Jaw', 'Teeth', 'Lips']]

def anchored_vwap(df, anchor_date):
    """
    Calculates the Anchored VWAP indicator for a given data frame.

    Parameters:
        -- df (pd.DataFrame): Data frame containing OHLCV (Open, High, Low, Close, Volume) data.
        -- anchor_date (str): Date to anchor the VWAP calculation. Format: 'YYYY-MM-DD'.

    Returns:
        -- pd.DataFrame: Data frame containing Anchored VWAP values for each row in the data frame.
    """
    # Filter data frame to rows after the anchor date
    df = df[df.index > anchor_date]

    # Calculate VWAP
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3  # Typical Price
    df['TPV'] = df['TP'] * df['Volume']  # Typical Price x Volume
    df['Cumulative_TPV'] = df['TPV'].cumsum()  # Cumulative Sum of Typical Price x Volume
    df['Cumulative_Volume'] = df['Volume'].cumsum()  # Cumulative Sum of Volume
    df['VWAP'] = df['Cumulative_TPV'] / df['Cumulative_Volume']  # VWAP

    return df[['VWAP']]

#6
def aroon(df, window=13):
    """
    Calculates the Aroon indicator for a given data frame.

    Parameters:
        -- df (pd.DataFrame): Data frame containing OHLC (Open, High, Low, Close) data.
        -- window (int): Number of periods to use for Aroon calculation.

    Returns:
        -- pd.DataFrame: Data frame containing Aroon Up and Aroon Down values for each row in the data frame.
    """
    high_max = df['High'].rolling(window=window+1).max()  # Highest High in the past window periods
    low_min = df['Low'].rolling(window=window+1).min()  # Lowest Low in the past window periods

    periods_since_high = window - df['High'].rolling(window=window+1).apply(lambda x: x.argmax(), raw=True).fillna(window).astype(int)
    periods_since_low = window - df['Low'].rolling(window=window+1).apply(lambda x: x.argmin(), raw=True).fillna(window).astype(int)

    aroon_up = (periods_since_high / window) * 100
    aroon_down = (periods_since_low / window) * 100

    return pd.DataFrame({'Aroon_Up': aroon_up, 'Aroon_Down': aroon_down})


#7
def aroon_oscillator(df, window=14):
    print("In aroon_oscillator function")
    """
    Calculates Aroon Oscillator for a given data frame.

    Parameters:
        -- df (pd.DataFrame): Data frame containing OHLC (Open, High, Low, Close) data.
        -- window (int): Window size for calculating Aroon Up and Aroon Down values. Default is 14.

    Returns:
        -- pd.DataFrame: A data frame containing Aroon Oscillator values for each row in the input data frame.
    """
    high_idx = df['High'].rolling(window=window).apply(lambda x: x.argmax(), raw=True)
    low_idx = df['Low'].rolling(window=window).apply(lambda x: x.argmin(), raw=True)
    # Calculate Aroon Up and Aroon Down values
    #high_idx = df['High'].rolling(window=window).idxmax()  # Index of highest high in the past window periods
    #low_idx = df['Low'].rolling(window=window).idxmin()  # Index of lowest low in the past window periods
    aroon_up = ((window - (df.index - high_idx)) / window) * 100
    aroon_down = ((window - (df.index - low_idx)) / window) * 100

    # Calculate Aroon Oscillator as the difference between Aroon Up and Aroon Down values
    aroon_oscillator = aroon_up - aroon_down

    # Create a data frame to store the Aroon Oscillator values
    aroon_df = pd.DataFrame({'Aroon_Up': aroon_up, 'Aroon_Down': aroon_down, 'Aroon_Oscillator': aroon_oscillator})

    return aroon_df['Aroon_Oscillator']

#8
#Check this out with the 2nd funtion
def average_true_range(df, period=14):
    # Calculate True Range
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = true_range.rolling(window=period).mean()
    
    return atr.rename('ATR2')

#9
def awesome_oscillator(df, period1=5, period2=34):
    # Calculate Simple Moving Averages
    sma1 = df['Close'].rolling(window=period1).mean()
    sma2 = df['Close'].rolling(window=period2).mean()
    
    # Calculate Awesome Oscillator
    ao = sma1 - sma2
    
    return ao.rename('AO')
#10
def balance_of_power(df):
    numerator = df['Close'] - df['Open']
    denominator = df['High'] - df['Low']
    bop = numerator / denominator * df['Volume']
    df['BOP'] = bop
    return df['BOP']

#11
def bollinger_percent_b(df, period=20, std=2):
    # Calculate Simple Moving Average and Standard Deviation
    sma = df['Close'].rolling(window=period).mean()
    std_dev = df['Close'].rolling(window=period).std()

    # Calculate Upper and Lower Bands
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)

    # Calculate %b
    percent_b = (df['Close'] - lower_band) / (upper_band - lower_band)

    return percent_b.rename('percent_b')

#12
def add_bollinger_bands(df, window=20, n_std=2):
    """
    Adds Bollinger Bands to a Pandas DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame containing a 'Close' column.
        window (int): Rolling window size for calculating the moving average and standard deviation.
        n_std (float): Number of standard deviations to use for the upper and lower bands.

    Returns:
        pandas.DataFrame: A new DataFrame with the Bollinger Bands added as columns.
    """
    close = df['Close']
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * n_std)
    lower_band = rolling_mean - (rolling_std * n_std)
    bb_percentage = (close - lower_band) / (upper_band - lower_band)
    
    new_df = df.copy()
    new_df['Bollinger Upper'] = upper_band
    new_df['Bollinger Lower'] = lower_band
    new_df['Bollinger %b'] = bb_percentage
    
    return new_df[['Bollinger Upper', 'Bollinger Lower']]

#13
def add_candlestick_patterns(df):
    """
    Adds candlestick patterns to a Pandas DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pandas.DataFrame: A new DataFrame with the detected candlestick patterns added as columns.
    """
    open_price = df['Open']
    high = df['High']
    low = df['Low']
    close = df['Close']
    pattern_list = []

    for i in range(len(df)):
        if i == 0:
            pattern_list.append('N/A')
            continue

        if close[i] > open_price[i]:
            body_direction = 'bull'
            upper_shadow = high[i] - close[i]
            lower_shadow = open_price[i] - low[i]
        else:
            body_direction = 'bear'
            upper_shadow = high[i] - open_price[i]
            lower_shadow = close[i] - low[i]

        body_length = abs(close[i] - open_price[i])
        total_length = body_length + upper_shadow + lower_shadow

        if body_direction == 'bull':
            if body_length / total_length > 0.6 and upper_shadow / body_length < 0.1 and lower_shadow / body_length < 0.4:
                pattern_list.append('Hammer')
            elif body_length / total_length > 0.6 and upper_shadow / body_length < 0.4 and lower_shadow / body_length < 0.1:
                pattern_list.append('Inverted Hammer')
            elif body_length / total_length > 0.7 and upper_shadow / body_length < 0.05 and lower_shadow / body_length > 0.2:
                pattern_list.append('Bullish Engulfing')
            elif body_length / total_length > 0.5 and upper_shadow / body_length < 0.2 and lower_shadow / body_length < 0.1:
                pattern_list.append('Piercing Line')
            else:
                pattern_list.append('N/A')
        elif body_direction == 'bear':
            if body_length / total_length > 0.6 and upper_shadow / body_length < 0.4 and lower_shadow / body_length < 0.1:
                pattern_list.append('Shooting Star')
            elif body_length / total_length > 0.6 and upper_shadow / body_length < 0.1 and lower_shadow / body_length < 0.4:
                pattern_list.append('Hanging Man')
            elif body_length / total_length > 0.7 and upper_shadow / body_length > 0.2 and lower_shadow / body_length < 0.05:
                pattern_list.append('Bearish Engulfing')
            elif body_length / total_length > 0.5 and upper_shadow / body_length < 0.1 and lower_shadow / body_length < 0.2:
                pattern_list.append('Dark Cloud Cover')
            else:
                pattern_list.append('N/A')

    df['Candlestick Pattern'] = pattern_list

    return df['Candlestick Pattern']

#14
def cog(df, period=14):
    """Calculate Center of Gravity (COG) for a DataFrame"""
    prices = (df['Close'] + df['High'] + df['Low'] + df['Open']) / 4
    cum_prices = prices.cumsum()
    cog_values = ((cum_prices.index + 1) * prices).rolling(period).sum() / cum_prices.rolling(period).sum()
    df['cog'] = cog_values
    return df['cog']

#15
def cpr(df):
    """Calculate Central Pivot Range (CPR) for a DataFrame"""
    pivot = (df['High'] + df['Low'] + df['Close']) / 3
    high_low = df['High'] - df['Low']
    upper_cpr = pivot + (high_low * 0.5)
    lower_cpr = pivot - (high_low * 0.5)
    mid_cpr = pivot + (high_low * 0.25)
    df['pivot'] = pivot
    df['upper_cpr'] = upper_cpr
    df['lower_cpr'] = lower_cpr
    df['mid_cpr'] = mid_cpr
    return df[['pivot', 'upper_cpr', 'lower_cpr', 'mid_cpr']]

#16
def cmf(df, period = 20):
    """Calculate Chaikin Money Flow (CMF) for a DataFrame"""
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_volume = mfm * df['Close'].shift(1)
    cmf_values = mf_volume.rolling(period).sum() / mfm.rolling(period).sum()
    df['cmf'] = cmf_values
    return df['cmf']
#17
def chv(df, period = 10):
    """Calculate Chaikin Volatility (CHV) for a DataFrame"""
    high_low_diff = df['High'] - df['Low']
    adl = ((2 * df['Close'] - df['Low'] - df['High']) / high_low_diff) * df['Volume']
    chv_values = adl.rolling(period).std() * np.sqrt(period)
    df['chv'] = chv_values
    return df['chv']

#18
def cfo(df, period = 10):
    """Calculate Chande Forecast Oscillator (CFO) for a DataFrame"""
    forecast = df['Close'].rolling(period).mean().shift(-period)
    cfo_values = 100 * (forecast - df['Close']) / df['Close']
    df['cfo'] = cfo_values
    return df['cfo']

#19
def cmo(df, period = 14):
    """Calculate Chande Momentum Oscillator (CMO) for a DataFrame"""
    diff = df['Close'].diff()
    up_sum = diff.where(diff > 0, 0).rolling(period).sum()
    down_sum = abs(diff.where(diff < 0, 0)).rolling(period).sum()
    cmo_values = 100 * (up_sum - down_sum) / (up_sum + down_sum)
    df['cmo'] = cmo_values
    return df['cmo']

#20
def ci(df, period = 14):
    """Calculate Choppiness Index (CI) for a DataFrame"""
    high_low_range = df['High'] - df['Low']
    atr = high_low_range.rolling(period).sum() / period
    max_high = df['High'].rolling(period).max()
    min_low = df['Low'].rolling(period).min()
    tr = max_high - min_low
    ci_values = 100 * np.log10(atr / tr) / np.log10(period)
    df['ci'] = ci_values
    return df['ci']

#21
def cci(df, period = 20):
    """Calculate Commodity Channel Index (CCI) for a DataFrame"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = typical_price.rolling(period).mean()
    mean_deviation = abs(typical_price - sma_tp).rolling(period).mean()
    cci_values = (typical_price - sma_tp) / (0.015 * mean_deviation)
    df['cci'] = cci_values
    return df['cci']

#22
def coppock_curve(df, long_roc_period = 30, short_roc_period = 14, wma_period = 20):
    """Calculate Coppock Curve for a DataFrame"""
    roc_long = ((df['Close'] / df['Close'].shift(long_roc_period)) - 1) * 100
    roc_short = ((df['Close'] / df['Close'].shift(short_roc_period)) - 1) * 100
    wma = ((wma_period * roc_short) + ((1 - wma_period) * roc_long)).rolling(wma_period).sum()
    coppock = wma.shift(wma_period)
    df['coppock'] = coppock
    return df['coppock']

#23 - returns 3
#24
def detrended_price_oscillator(df, period = 25):
    """Calculate Detrended Price Oscillator for a DataFrame"""
    dpo = pd.Series(np.zeros(len(df.index)))
    dpo.index = df.index
    ma = df['Close'].rolling(window=period).mean().shift(int(period / 2) + 1)
    for i in range(int(period / 2) + 1, len(df.index)):
        dpo[i] = df['Close'][i] - ma[i]
    df['dpo'] = dpo
    return df['dpo']

#25
def disparity_index(df, period = 40):
    """Calculate Disparity Index for a DataFrame"""
    ma = df['Close'].rolling(window=period).mean()
    di = df['Close'] / ma * 100
    df['disparity_index'] = di
    return df['disparity_index']

#25
def donchian_channel(df, period = 75):
    """Calculate Donchian Channel for a DataFrame"""
    high = df['High'].rolling(window=period).max()
    low = df['Low'].rolling(window=period).min()
    upper_dc = high.shift(period-1)
    lower_dc = low.shift(period-1)
    df['upper_dc'] = upper_dc
    df['lower_dc'] = lower_dc
    return df[['upper_dc', 'lower_dc']]

#26
def ehlers_fisher_transform(df, period = 13):
    """
    Calculate Ehlers Fisher Transform for a DataFrame
    """
    price = (df['High'] + df['Low'] + df['Close']) / 3
    x = price.diff().fillna(0)
    period_half = int(period / 2)
    alpha = x.ewm(span=period_half, min_periods=period_half).var()
    alpha += alpha.shift(period_half)
    alpha = alpha.fillna(alpha.mean())
    efi = (x - x.ewm(alpha=1 / period, min_periods=period).mean()) / np.sqrt(alpha)
    df['efi'] = efi
    return df['efi']

#27
def elder_impulse_system(df):
    """Calculate Elder Impulse System for a DataFrame"""
    ema_13 = df['Close'].ewm(span=13).mean() # 13-day EMA
    macd = MACD(df['Close'], window_fast=12, window_slow=26, window_sign=9).macd() # MACD histogram
    bullish = (ema_13 > ema_13.shift(1)) & (macd > macd.shift(1)) # bullish condition
    bearish = (ema_13 < ema_13.shift(1)) & (macd < macd.shift(1)) # bearish condition
    df['elder_bullish'] = np.where(bullish, 1, 0) # 1 if bullish, 0 otherwise
    df['elder_bearish'] = np.where(bearish, 1, 0) # 1 if bearish, 0 otherwise
    return df[['elder_bullish', 'elder_bearish']]

#28 - returns 2
def elder_ray_index(df, period=13):
    """
    Calculate the Elder Ray Index for a given DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing OHLC data.
    period (int): Number of periods to use for calculation.
    
    Returns:
    pd.DataFrame: DataFrame with Elder Ray Index values.
    """
    high_max = df['High'].rolling(window=period).max()
    low_min = df['Low'].rolling(window=period).min()
    ema_13 = df['Close'].ewm(span=period).mean()
    
    bull_power = high_max - ema_13
    bear_power = ema_13 - low_min
    
    elder_ray_index = pd.DataFrame({'Bull Power': bull_power, 'Bear Power': bear_power})
    
    return elder_ray_index

#29 - returns 3
def fractal_chaos_bands(df, period=13, deviation=3):
    """Calculate Fractal Chaos Bands for a DataFrame"""
    df["High_High"] = df['High'].rolling(period).max().shift(period-1)
    df["Low_Low"] = df['Low'].rolling(period).min().shift(period-1)
    df["HH_LC"] = df["High_High"] - df['Close']
    df["HC_LL"] = df['Close'] - df["Low_Low"]
    df["Median_FCB"] = (df["HH_LC"] + df["HC_LL"]) / 2
    df["FCB_Upper"] = df["Median_FCB"] + deviation * df["Median_FCB"].rolling(period).std()
    df["FCB_Lower"] = df["Median_FCB"] - deviation * df["Median_FCB"].rolling(period).std()
    df.drop(["High_High", "Low_Low", "HH_LC", "HC_LL"], axis=1, inplace=True)
    return df[['Median_FCB', "FCB_Upper", "FCB_Lower"]]

#30
def fractal_chaos_oscillator(df, period=13):
    """
    Calculate the Fractal Chaos Oscillator for a DataFrame.
    """
    prices = df['Close'].to_numpy()
    highs = df['High'].to_numpy()
    lows = df['Low'].to_numpy()
    fractal_dimension = np.zeros(len(prices))

    for i in range(period, len(prices)):
        max_high = max(highs[i - period: i])
        min_low = min(lows[i - period: i])

        price_range = max_high - min_low
        if price_range == 0:
            price_range = 0.0001

        fractal_dimension[i] = np.log(price_range) / np.log(period)

    reference_value = np.full_like(fractal_dimension, np.mean(fractal_dimension))
    fco = fractal_dimension - reference_value

    df['fco'] = fco
    return df['fco']

#31
def gator_oscillator(df, jaw_period=13, teeth_period=8, lips_period=5, offset=8):
    """
    Calculates the Gator Oscillator for a DataFrame

    Parameters:
    df (pd.DataFrame): input data
    jaw_period (int): period for the jaw line (default=13)
    teeth_period (int): period for the teeth line (default=8)
    lips_period (int): period for the lips line (default=5)
    offset (int): offset for the upper and lower lines (default=8)

    Returns:
    pd.DataFrame: a DataFrame with the Gator Oscillator values as a new column
    """
    jaw = df['Close'].rolling(window=jaw_period).mean().shift(offset)
    teeth = df['Close'].rolling(window=teeth_period).mean().shift(offset)
    lips = df['Close'].rolling(window=lips_period).mean()
    upper = jaw.subtract(teeth).abs().rolling(window=offset).mean().shift(offset)
    lower = jaw.subtract(lips).abs().rolling(window=offset).mean()
    df['gator'] = upper.subtract(lower)
    return df['gator']

#32
def gopalakrishnan_range_index(df, period = 14):
    """
    Computes the Gopalakrishnan Range Index for a given DataFrame and period
    
    Parameters:
    df (pandas.DataFrame): A pandas DataFrame containing the OHLC data
    period (int): The period for which the Gopalakrishnan Range Index needs to be calculated
    
    Returns:
    pandas.DataFrame: A DataFrame containing the Gopalakrishnan Range Index values as a new column named 'GRI'
    """
    high_low_range = df['High'] - df['Low']
    high_previous_close_range = abs(df['High'] - df['Close'].shift())
    low_previous_close_range = abs(df['Low'] - df['Close'].shift())
    true_range = high_low_range.combine(high_previous_close_range, max).combine(low_previous_close_range, max)
    gopala_range_index = true_range.rolling(window=period).sum() / (period * df['Close'])
    df['GRI'] = gopala_range_index
    return df['GRI']

#33 - returns 3
def guppy_moving_average(df, short_periods=50, medium_periods=100, long_periods=200):
    """Calculate Guppy Multiple Moving Averages for a DataFrame"""
    df[f'short_ma_{short_periods}'] = df['Close'].rolling(short_periods).mean()
    df[f'medium_ma_{medium_periods}'] = df['Close'].rolling(medium_periods).mean()
    df[f'long_ma_{long_periods}'] = df['Close'].rolling(long_periods).mean()
    return df[[f'short_ma_{short_periods}', f'medium_ma_{medium_periods}', f'long_ma_{long_periods}']]

#34
def high_minus_low(df, period=14):
    high = df['High'].rolling(window=period).max()
    low = df['Low'].rolling(window=period).min()
    hml = high - low
    return hml.rename('HML')

#35

def highest_high(df, period=14):
    hh = df['High'].rolling(period).max()
    return hh.rename('Highest_High')

# Example usage:
# df = pd.read_csv('your_data.csv')
# hh_df = highest_high(df)
# print(hh_df)

#36
def historical_volatility(df, period=14):
    """
    Calculates the historical volatility of a security over a given period of time.
    Input: 
        - df: pandas DataFrame with 'High' and 'Low' columns
        - period: integer, the number of days to consider for calculating volatility
    Output:
        - pandas DataFrame with 'HV' column representing the historical volatility
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    log_returns = np.log(close / close.shift(1))
    HV = log_returns.rolling(period).std() * np.sqrt(252)
    return HV.rename('HV')

#37 - returns 5
def ichimoku_clouds(df, tenkan_period=9, kijun_period=26, senkou_period=52):
    """
    Calculate the Ichimoku Clouds technical indicator for a given DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing columns for 'High', 'Low', and 'Close' prices.
    tenkan_period : int
        Period used to calculate Tenkan-sen (conversion line).
        Default is 9.
    kijun_period : int
        Period used to calculate Kijun-sen (base line).
        Default is 26.
    senkou_period : int
        Period used to calculate Senkou Span A and B (leading spans).
        Default is 52.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the following columns:
        - tenkan_sen
        - kijun_sen
        - senkou_span_a
        - senkou_span_b
        - chikou_span
    """
    # Calculate Tenkan-sen (Conversion Line)
    tenkan_high = df['High'].rolling(window=tenkan_period).max()
    tenkan_low = df['Low'].rolling(window=tenkan_period).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Calculate Kijun-sen (Base Line)
    kijun_high = df['High'].rolling(window=kijun_period).max()
    kijun_low = df['Low'].rolling(window=kijun_period).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Calculate Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
    
    # Calculate Senkou Span B (Leading Span B)
    senkou_high = df['High'].rolling(window=senkou_period).max()
    senkou_low = df['Low'].rolling(window=senkou_period).min()
    senkou_span_b = ((senkou_high + senkou_low) / 2).shift(kijun_period)
    
    # Calculate Chikou Span (Lagging Span)
    chikou_span = df['Close'].shift(-kijun_period)
    
    # Combine all columns into a DataFrame and return
    ichimoku_df = pd.DataFrame({
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    })
    
    return ichimoku_df

#38
def intraday_momentum_index(df, period=14):
    prev_close = df['Close'].shift(1)
    upward_price_change = df['High'] - prev_close
    downward_price_change = prev_close - df['Low']
    total_price_change = upward_price_change + downward_price_change
    
    positive_price_change = np.where(upward_price_change > downward_price_change, upward_price_change, 0)
    negative_price_change = np.where(downward_price_change > upward_price_change, downward_price_change, 0)
    
    sum_positive_price_change = pd.Series(positive_price_change).rolling(window=period).sum()
    sum_negative_price_change = pd.Series(negative_price_change).rolling(window=period).sum()
    
    imi = 100 * (sum_positive_price_change / total_price_change)
    df['IMI'] = imi
    return df['IMI']

#39 - returns 3
def keltner_channel(df, period=20, multiplier=2):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    ema = typical_price.ewm(span=period, adjust=False).mean()
    atr = df['High'] - df['Low']
    atr = atr.rolling(period).mean()
    upper_band = ema + multiplier * atr
    lower_band = ema - multiplier * atr
    keltner_channel = pd.concat([ema, upper_band, lower_band], axis=1)
    keltner_channel.columns = ['Keltner Center', 'Keltner Upper', 'Keltner Lower']
    return keltner_channel

#40 - returns 
def linear_reg_forecast(df, period = 30):
    """
    Computes the Linear Regression Forecast (LRF) for a given period.

    Parameters:
    df (pandas.DataFrame): The input data.
    period (int): The number of periods to forecast.

    Returns:
    pandas.Series: A Series containing the LRF values.
    """

    # Compute the linear regression coefficients
    lr = LinearRegression()
    x = pd.Series(range(1, len(df) + 1))
    x = x.to_numpy().reshape(-1, 1)
    y = df['Close'].to_numpy()
    lr.fit(x, y)

    # Compute the LRF values
    lrf = pd.Series(lr.predict(x), index=df.index)

    # Compute the forecast values
    forecast = pd.Series(lr.predict(pd.Series(range(len(df), len(df) + period)).to_numpy().reshape(-1, 1)),
                         index=pd.date_range(start=df.index[-1], periods=period + 1, freq='B')[1:])

    # Combine the LRF and forecast values into a single Series
    result = pd.concat([lrf, forecast], axis=0)

    # Rename the Series
    result.name = 'LRF'

    return result

'''
#41
from sklearn.linear_model import LinearRegression

def linear_reg_intercept(df, column_name = 'Close'):
    # Select the column to be used as the dependent variable
    y = df[column_name].values.reshape(-1, 1)
    
    # Create a column of integers as the independent variable
    x = df.reset_index().index.values.reshape(-1, 1)
    
    # Fit the linear regression model to the data
    model = LinearRegression().fit(x, y)
    
    # Extract the intercept value from the model
    intercept = model.intercept_[0]
    
    # Create a new data frame with the intercept value as the only column
    result_df = pd.DataFrame({'Linear_Reg_Intercept': [intercept]})
    
    return result_df

#42
from sklearn.linear_model import LinearRegression

def linear_reg_r2(df, target_col, feature_col):
    # Create a copy of the original dataframe
    df = df.copy()
    
    # Calculate the target variable percentage change
    df[target_col + '_pct'] = df[target_col].pct_change()
    
    # Calculate the feature variable percentage change
    df[feature_col + '_pct'] = df[feature_col].pct_change()
    
    # Drop the NaN values
    df.dropna(inplace=True)
    
    # Fit the linear regression model
    X = df[feature_col + '_pct'].values.reshape(-1, 1)
    y = df[target_col + '_pct'].values.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    
    # Calculate the R-squared score
    r2 = reg.score(X, y)
    
    # Return the R-squared score as a dataframe
    return pd.DataFrame({'Linear Reg R2': [r2]})

'''
#41
def linear_reg_slope(df, period = 30):
    """
    Calculates the linear regression slope for a given data frame and period.

    Args:
    df (pandas.DataFrame): Data frame containing the 'Close' prices.
    period (int): Period for which the slope is to be calculated.

    Returns:
    pandas.DataFrame: Data frame containing the linear regression slope for each period.
    """

    # Calculate the slope for each period
    slope_list = []
    for i in range(period, len(df)):
        x = np.array(range(period))
        y = np.array(df['Close'][i-period:i])
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        lr = LinearRegression().fit(x, y)
        slope_list.append(lr.coef_[0][0])

    # Create a new data frame with the slope values
    df_slope = pd.DataFrame(slope_list, index=df.index[period:], columns=['slope'])

    return df_slope['slope']

#42
def lowest_low(df):
    # Extract the "Low" column from the dataframe
    low_values = df['Low']
    
    # Find the index of the lowest low value
    min_index = low_values.idxmin()
    
    # Extract the row with the lowest low value
    lowest_low_row = df.loc[[min_index]]
    
    # Return the dataframe with the row containing the lowest low value
    return lowest_low_row   

#43 - return 3
def macd(df, n_fast=12, n_slow=26, n_signal=9):
    """
    Computes the Moving Average Convergence Divergence (MACD) of a given time series.

    Parameters:
    df (pandas.DataFrame): The input data.
    n_fast (int): The number of periods for the fast EMA.
    n_slow (int): The number of periods for the slow EMA.
    n_signal (int): The number of periods for the signal line.

    Returns:
    pandas.DataFrame: A DataFrame containing the MACD values, signal line, and histogram.
    """

    # Compute the fast and slow exponential moving averages
    ema_fast = df['Close'].ewm(span=n_fast, min_periods=n_fast).mean()
    ema_slow = df['Close'].ewm(span=n_slow, min_periods=n_slow).mean()

    # Compute the MACD line and signal line
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=n_signal, min_periods=n_signal).mean()

    # Compute the histogram
    histogram = macd_line - signal_line

    # Combine the MACD, signal line, and histogram into a single DataFrame
    result = pd.concat([macd_line, signal_line, histogram], axis=1)

    # Rename the columns
    result.columns = ['MACD', 'Signal', 'Histogram']

    return result[['MACD', 'Signal', 'Histogram']]

def macd_divergence(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Computes the MACD Divergence and Histogram for a given DataFrame.

    Parameters:
    df (pandas.DataFrame): The input data.
    fast_period (int): The fast moving average period.
    slow_period (int): The slow moving average period.
    signal_period (int): The signal line period.

    Returns:
    pandas.DataFrame: A DataFrame containing the MACD Divergence and Histogram values.
    """
    # Compute the exponential moving averages for the fast and slow periods
    fast_ema = df['Close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['Close'].ewm(span=slow_period, adjust=False).mean()

    # Compute the MACD line and the signal line
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Compute the MACD Divergence
    macd_divergence = macd_line - signal_line

    # Compute the MACD Histogram
    macd_histogram = macd_line - signal_line

    # Create a DataFrame containing the MACD Divergence and Histogram values
    result = pd.DataFrame({'MACD Divergence': macd_divergence, 'MACD Histogram': macd_histogram})

    return result

#45
def mass_index(df, ema_period=9, ema_of_ema_period=9):
    """
    Calculates the Mass Index based on a given DataFrame.

    Parameters:
    df (pandas.DataFrame): A DataFrame with columns 'High' and 'Low'.
    ema_period (int): The period used to calculate the Exponential Moving Average (EMA) of the high-low range.
    ema_of_ema_period (int): The period used to calculate the EMA of the EMA of the high-low range.

    Returns:
    pandas.DataFrame: A DataFrame with one column 'Mass_Index'.
    """

    # Calculate the high-low range
    df['range'] = df['High'] - df['Low']

    # Calculate the Exponential Moving Average (EMA) of the high-low range
    df['ema'] = df['range'].ewm(span=ema_period, min_periods=ema_period).mean()

    # Calculate the EMA of the EMA of the high-low range
    df['ema_of_ema'] = df['ema'].ewm(span=ema_of_ema_period, min_periods=ema_of_ema_period).mean()

    # Calculate the Mass Index
    df['Mass_Index'] = df['ema'] / df['ema_of_ema']

    # Return a DataFrame with only the Mass Index values
    return df['Mass_Index']

#46
def median_price(df):
    """
    Computes the Median Price for a given DataFrame.

    Parameters:
    df (pandas.DataFrame): The input data.

    Returns:
    pandas.DataFrame: A DataFrame containing the Median Price values.
    """

    # Compute the median price
    median_price = (df['High'] + df['Low']) / 2
    median_price.name = 'Median Price'
    # Return the result as a DataFrame
    return median_price

#47
def momentum_indicator(df, n=14):
    """
    Computes the Momentum Indicator for a given DataFrame.

    Parameters:
    df (pandas.DataFrame): The input data.
    n (int): The number of periods to use for the calculation.

    Returns:
    pandas.DataFrame: A DataFrame containing the Momentum Indicator values.
    """

    # Compute the momentum indicator
    momentum = df['Close'].diff(n)
    momentum.name = 'Momentum_Indicator'
    # Return the result as a DataFrame
    return momentum

#48
def money_flow_index(df, n=14):
    """
    Computes the Money Flow Index (MFI) for a given DataFrame.

    Parameters:
    df (pandas.DataFrame): The input data.
    n (int): The number of periods to use for the calculation.

    Returns:
    pandas.DataFrame: A DataFrame containing the MFI values.
    """
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    pmf = pd.Series(0, index=df.index)
    nmf = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if tp[i] > tp[i-1]:
            pmf[i] = tp[i] * max(df['High'][i] - tp[i], 0)
        elif tp[i] < tp[i-1]:
            nmf[i] = tp[i] * max(tp[i] - df['Low'][i], 0)
    mfr = pmf.rolling(window=n).sum() / nmf.rolling(window=n).sum()
    mfi = 100 - (100 / (1 + mfr))
    return mfi.rename('Money Flow Index')









