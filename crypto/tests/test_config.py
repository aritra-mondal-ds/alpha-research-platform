import ta
import pandas as pd
import numpy as np

# Quick smoke test on dummy data
df = pd.DataFrame({
    'open':   [100.0]*50,
    'high':   [105.0]*50,
    'low':    [95.0]*50,
    'close':  [101.0 + i*0.1 for i in range(50)],
    'volume': [1000.0]*50
})

rsi = ta.momentum.rsi(df['close'], window=14)
ema = ta.trend.ema_indicator(df['close'], window=9)
print('ta library OK')
print('RSI sample:', round(rsi.dropna().iloc[-1], 2))
print('EMA sample:', round(ema.dropna().iloc[-1], 2))
