Building Neural Networks
====================================================================

In this work, we implemented the neural networks utilized in the novel high-frequency trading algorithm and developed a real-world data trading simulator to test the profitability of them.

Previous Work
------------

Our project lies in a recent research proposed by __Angus Parsonson__ (see the paper in _[Building a High-Frequency Trading Algorithm Using An Ordinary Differential Equation Recurrent Neural Network]( https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3956198)_).

Guidelines
------------

1. Execute _train_ file in **simulator/traders/** to train deep learning models. General checkpoints will be saved in **simulator/traders/checkpoints/**.
2. Execute _predict_ file in **simulator/utils/** to prepare for backtesting the hybridized strategies conducted in **grid_trading_algorithm_using_deep_learning**.
3. Execute _simulator_ file in **simulator** to simulate an algorithmic trading strategy in real-world scenarios.
