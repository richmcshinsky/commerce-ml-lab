"""Demand forecasting module for Project 01.

Models inventory demand from M5 (Walmart) retail data and translates
forecasts into actionable reorder decisions.

Modules
-------
data
    M5 data loading, melting wide-to-long, and train/test splitting.
features
    Lag features, rolling statistics, calendar and price features.
models
    Baseline (naive, seasonal naive, MA), ETS, and LightGBM global model.
inventory
    Newsvendor model, safety stock, and (s, S) reorder policy simulation.
evaluate
    Walk-forward backtesting and metric summary tables.
"""
