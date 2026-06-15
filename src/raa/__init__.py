"""Regime-Aware Strategic Asset Allocation research framework.

The package is organised into focused sub-packages:

- ``raa.utils``      configuration, logging, I/O and plotting helpers
- ``raa.data``       FMP client, macro + price collection, asset universe
- ``raa.regimes``    regime detection (rule-based, HMM, clustering, market-implied)
- ``raa.analysis``   performance metrics and per-regime statistics
- ``raa.portfolio``  static and regime-aware portfolio construction + backtesting
- ``raa.reporting``  figure and table generation for the report
"""

__version__ = "0.1.0"
