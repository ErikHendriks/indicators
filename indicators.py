#!/usr/bin/env python3

import numpy as np
import pandas as pd

from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from pandas.io.json import json_normalize

conf = [line.strip('\n') for line in open('/path/to/file.conf')]
accountID = conf[0]
api = API(access_token = conf[1],\
          environment = conf[2])

ohlcd = {'count': 200,'granularity': 'H1'}
symbol = 'EUR_USD'
candle = InstrumentsCandles(instrument=symbol,params=ohlcd)
api.request(candle)

prices = pd.DataFrame.from_dict(json_normalize(candle.response['candles']))
prices.time = pd.to_datetime(prices.time)
prices = prices.set_index(prices.time)
prices.columns = [['complete','close','high','low','open','time','volume']]

for column in ['close','high','low','open']:
    prices[column] = prices[column].astype(float)

class indicator():

    def bollinger(prices,periods,deviations):
        """
        Bollinger Bands

        prices: dataframe of OHLC data
        periods: list of periods to compute bollinger bands
        deviations: deviations to use to calculate bands(upper & lower)
        return: bollinger bands
        """

        results = indicator()
        boll = {}

        for i in range(0,len(periods)):

            mid = prices.close.rolling(periods[i]).mean()
            std = prices.close.rolling(periods[i]).std()

            upper = mid+deviations*std
            lower = mid-deviations*std

            df = pd.concat((upper,mid,lower),axis=1)
            df.columns = [['upper','mid','lower']]

            boll[periods[i]] = df

        results.bands = boll

        return results

    def cci(prices,periods):
        """
        CCI (Commodity Channel Index)

        prices: dataframe of OHLC data
        periods: list of periods to compute indicator
        return: MACD for given periods
        """

        results = indicator()
        CCI = {}

        for i in range(0,len(periods)):

            MA = prices.close.rolling(periods[i]).mean()
            std = prices.close.rolling(periods[i]).std()

            D = (prices.close-MA)/std

            CCI[periods[i]] = pd.DataFrame((prices.close-MA)/(0.015*D))
            CCI[periods[i]].columns = [['close']]

        results.cci = CCI

        return results

    def macd(prices,periods):
        """
        Moving Average Convergence Divergence

        prices: dataframe of OHLC data
        periods: 1x2 array of EMA values
        return: MACD for given periods
        """

        results = indicator()

        EMA1 = prices.close.ewm(span=periods[0]).mean()
        EMA2 = prices.close.ewm(span=periods[1]).mean()

        MACD = pd.DataFrame(EMA1-EMA2)
        MACD.columns = [['SL']]

        SigMACD = MACD.rolling(3).mean()
        SigMACD.columns = [['SL']]

        results.line = MACD
        results.signal = SigMACD

        return results

    def momentum(prices,periods):
        """
        prices: dataframe of OHLC data
        periods: list of periods to calculate function value
        return: momentum indicator
        """

        results = indicator()
        opn = {}
        close = {}

        for i in range(0,len(periods)):

            opn[periods[i]] = pd.DataFrame(prices.open.iloc[periods[i]:] - prices.open.iloc[:-periods[i]].values, index=prices.iloc[periods[i]:].index)
            close[periods[i]] = pd.DataFrame(prices.close.iloc[periods[i]:] - prices.close.iloc[:-periods[i]].values, index=prices.iloc[periods[i]:].index)

            opn[periods[i]].columns = [['open']]
            close[periods[i]].columns = [['close']]

        results.opn = opn
        results.close = close

        return results

    def movingAverage(prices,periods):
        """
        prices: dataframe of OHLC data
        periods: list of periods to compute indicator values
        return: dataframe moving average on close prices
        """

        return prices.close.rolling(center=False,window=periods[0]).mean()


    def paverage(prices,periods):
        """
        Price Average

        prices: dataframe of OHLC data
        periods: list of periods to compute indicator values
        return: averages over the given periods
        """

        results = indicator()
        avs = {}

        for i in range(0,len(periods)):

            avs[periods[i]] = pd.DataFrame(prices[['open','high','low','close']].rolling(periods[i]).mean())


        results.avs = avs

        return results



    def proc(prices,periods):
        """
        Price Rate Of Change

        prices: dataframe of OHLC data
        periods: list of periods to calculate function value
        return: PROC values for indicated periods
        """

        results = indicator()
        proc = {}

        for i in range(0,len(periods)):

            proc[periods[i]] = pd.DataFrame((prices.close.iloc[periods[i]:]-prices.close.iloc[:-periods[i]].values)/prices.close.iloc[:-periods[i]].values)
            proc[periods[i]].columns = [['close']]

        results.proc = proc

        return results

    def williams(prices,periods):
        """
        Williams %R

        prices: dataframe of OHLC data
        periods: list of periods to calculate function value
        return: wiliams oscillator function values
        """

        results = indicator()
        close = {}

        for i in range(0,len(periods)):

            Rs = []

            for j in range(periods[i],len(prices)-periods[i]):

                C = prices.close.iloc[j+1]
                H = float(prices.high.iloc[j-periods[i]:j].max())
                L = float(prices.low.iloc[j-periods[i]:j].min())

                if H == L:

                    R = 0

                else:

                    R = -100*(H-C)/(H-L)

                Rs = np.append(Rs,R)

            df = pd.DataFrame(Rs,index=prices.iloc[periods[i]+1:-periods[i]+1].index)
            df.columns = [['R']]
            df = df.dropna()

            close[periods[i]] = df

        results.close = close

        return results

"""
## Create lists of each period required by the functions
"""
bollingerKey = [15]
cciKey = [15]
macdKey = [15,30]
momentumKey = [3,4,5,8,9,10]
movingAverageKey = [30]
paverageKey = [2]
procKey = [12,13,14,15]
williamsKey = [6,7,8,9,10]

#bollingerDict = indicator.bollinger(prices,bollingerKey,2)
#cciDict = indicator.cci(prices,cciKey)
#macdDict = indicator.macd(prices,macdKey)
#momentumDict = indicator.momentum(prices,momentumKey)
#movingAverageDict = indicator.movingAverage(prices,movingAverageKey)
#paverageDict = indicator.paverage(prices,paverageKey)
#procDict = indicator.proc(prices,procKey)
#williamsDict = indicator.williams(prices,williamsKey)
