
High-frequency Trading Backtesting in Python
====================================================================

In this work, we focus on developing high-frequency trading strategies. The **HftBacktest** framework is selected on the basis of its accuracy, in terms of market replay-based backtesting based on full order book and trade tick feed data.

Documentation
------------
_See [full document here](https://hftbacktest.readthedocs.io/)_.

Guidelines
------------
1. Collect cryptocurrency exchanges into limit order book data from _Binance Futures_ using an implementation in [collect-binancefutures](https://github.com/nkaz001/collect-binancefutures.git).
2. Process collected data into _npz_ and _csv_ files by executing _data_preparing_ and _data_processing_ files. Backtesting is used for the former, while optimizing deep neural networks is used for the latter.
3. Build deep learning models implemented in **deep_learning** for the hybrid grid trading strategies. 
4. Backtest a variety of implemented grid trading strategies in _grid_trading_origin_, _grid_trading_with_support_resistance_, and _hybrid_grid_trading_with_support_resistance_ files.