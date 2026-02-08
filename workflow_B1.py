import argparse
import math
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from tqcenter import tq


WINDOW = 50
KDJ_N = 9
KDJ_ALPHA = 1 / 3

FACTOR_SET_5 = [
    "VAM50",
    "REV50",
    "LVOL50",
    "PVDC50",
    "MSI50",
]

FACTOR_SET_7 = [
    "BreakoutFollow_50",
    "DrawdownPressure_50",
    "VolatilityExpansion_50",
    "PriceVolumeCoMove_50",
    "UpperWickRatio_50",
    "OpenCloseDominance_50",
    "VolumeImpulse_50",
]

FACTOR_SET_10 = [
    "Momentum_Breakthrough_Combo",
    "Volume_Contraction_Reversal",
    "Volatility_Squeeze_Breakout",
    "Amount_Concentration_Ratio",
    "Liquidity_Shock_Adaptive",
    "Volume_Cluster_Ratio",
    "Price_Squeeze_Indicator",
    "VWAP_Deviation",
    "Trend_Strength_Composite",
    "Support_Resistance_Break",
]

FACTOR_SET_6 = [
    "Bottom_Volume_Reversal",
    "Consolidation_Breakout_Pullback",
    "N_Pattern_Pullback",
    "Triple_Pattern_Capture",
    "Volume_Price_Health_Score",
    "Pullback_Depth_Volume_Match",
]

ALL_FACTORS = FACTOR_SET_5 + FACTOR_SET_7 + FACTOR_SET_10 + FACTOR_SET_6


def chunk_list(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_ohlcv(
    stock_list: List[str],
    count: int,
    end_time: str,
) -> Dict[str, pd.DataFrame]:
    fields = ["Open", "High", "Low", "Close", "Volume"]
    field_frames: Dict[str, List[pd.DataFrame]] = {f: [] for f in fields}

    chunk_size = max(1, 24000 // max(1, count))
    for chunk in chunk_list(stock_list, chunk_size):
        data = tq.get_market_data(
            field_list=fields,
            stock_list=chunk,
            start_time="",
            end_time=end_time,
            count=count,
            dividend_type="front",
            period="1d",
            fill_data=True,
        )
        for field in fields:
            df_field = tq.price_df(data, field, column_names=chunk)
            field_frames[field].append(df_field)

    result: Dict[str, pd.DataFrame] = {}
    for field in fields:
        if not field_frames[field]:
            result[field] = pd.DataFrame()
        else:
            df = pd.concat(field_frames[field], axis=1)
            df = df.sort_index()
            result[field] = df
    return result


def compute_kdj_j(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    low_n = low.rolling(KDJ_N, min_periods=KDJ_N).min()
    high_n = high.rolling(KDJ_N, min_periods=KDJ_N).max()
    rsv = (close - low_n) / (high_n - low_n) * 100
    k = rsv.ewm(alpha=KDJ_ALPHA, adjust=False).mean()
    d = k.ewm(alpha=KDJ_ALPHA, adjust=False).mean()
    j = 3 * k - 2 * d
    return j


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = compute_atr(high, low, close, 1)
    tr_smooth = tr.rolling(period, min_periods=period).sum()
    pos_dm_smooth = pd.Series(pos_dm, index=high.index).rolling(period, min_periods=period).sum()
    neg_dm_smooth = pd.Series(neg_dm, index=high.index).rolling(period, min_periods=period).sum()
    pos_di = 100 * pos_dm_smooth / tr_smooth.replace(0, np.nan)
    neg_di = 100 * neg_dm_smooth / tr_smooth.replace(0, np.nan)
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan)
    return dx.rolling(period, min_periods=period).mean()


def compute_factors_for_stock(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    vol: pd.Series,
) -> Dict[str, float]:
    min_required_length = max(WINDOW, 60)
    if len(close) < min_required_length:
        return {name: np.nan for name in ALL_FACTORS}
    ret = close.pct_change(fill_method=None)
    ret10 = ret.rolling(10).sum()
    ret20 = ret.rolling(20).sum()
    std50 = ret.rolling(WINDOW, min_periods=WINDOW).std(ddof=0)
    vol_ma50 = vol.rolling(WINDOW, min_periods=WINDOW).mean()
    range_den = (high - low).replace(0, np.nan)

    prev_high_50 = high.shift(1).rolling(WINDOW, min_periods=WINDOW).max()
    max_close_50 = close.rolling(WINDOW, min_periods=WINDOW).max()

    vam50 = rolling_zscore(vol, WINDOW) * rolling_zscore(ret10, WINDOW)
    rev50 = -rolling_zscore(ret20, WINDOW)
    lvol50 = -std50
    pvdc50 = -ret.rolling(WINDOW, min_periods=WINDOW).corr(vol.pct_change(fill_method=None))
    msi50 = rolling_zscore(((high - low) / close) * vol, WINDOW)

    breakout_follow_50 = ((close - prev_high_50) / prev_high_50) * (vol / vol_ma50)
    drawdown_pressure_50 = (max_close_50 - close) / max_close_50
    volatility_expansion_50 = std50.diff()
    price_volume_comove_50 = close.diff().rolling(WINDOW, min_periods=WINDOW).corr(vol.diff())
    upper_wick_ratio_50 = ((high - np.maximum(close, open_)) / range_den).rolling(
        WINDOW, min_periods=WINDOW
    ).mean()
    open_close_dominance_50 = ((close - open_) / range_den).rolling(
        WINDOW, min_periods=WINDOW
    ).mean()
    volume_impulse_50 = vol / vol_ma50

    amount = close * vol
    amount_ma20 = amount.rolling(20).mean()
    amount_ma60 = amount.rolling(60).mean()
    amount_rank = amount.rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    recent_amount_sum = amount.rolling(5).sum().shift(1)

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    atr14 = compute_atr(high, low, close, 14)
    atr20 = compute_atr(high, low, close, 20)
    rsi6 = compute_rsi(close, 6)
    rsi14 = compute_rsi(close, 14)

    recent_high_5 = high.rolling(5).max().shift(1)
    recent_low_5 = low.rolling(5).min()
    ma5_20_sign = np.sign(ma5 - ma20)

    volume_ma10 = vol.rolling(10).mean()
    volume_ma5 = vol.rolling(5).mean()
    volume_ma20 = vol.rolling(20).mean()
    volume_ratio10 = vol / volume_ma10
    volume_ratio20 = vol / volume_ma20

    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    close_ma60 = close.rolling(60).mean()

    open_safe = open_.replace(0, np.nan)
    candle_body = (close - open_) / open_safe
    upper_shadow = (high - np.maximum(close, open_)) / open_safe
    lower_shadow = (np.minimum(close, open_) - low) / open_safe

    price_break = close / recent_high_5.replace(0, np.nan)
    trend_dir = np.sign(close - ma20)
    momentum_break = volume_ratio10 * price_break * (rsi6 / 50.0) * trend_dir

    volume_contraction = 1 - (vol / volume_ma5)
    rebound_norm = (close - recent_low_5) / atr14.replace(0, np.nan)
    extreme_reversal = volume_contraction * rebound_norm * ma5_20_sign

    upper_band = ma20 + 2 * std20
    lower_band = ma20 - 2 * std20
    bb_width = (upper_band - lower_band) / ma20.replace(0, np.nan)
    vol_ratio = bb_width / (atr14 / close.replace(0, np.nan))
    price_zscore = (close - ma20) / std20.replace(0, np.nan)
    volume_signal = np.sign(vol - volume_ma10)
    vol_regime = vol_ratio * price_zscore * volume_signal

    amount_concentration = amount / recent_amount_sum.replace(0, np.nan)
    amount_strength = amount / amount_ma20.replace(0, np.nan)
    price_change = close.pct_change(3)
    amount_change = amount.pct_change(3)
    price_amount_alignment = np.sign(price_change) * np.sign(amount_change)
    amount_concentration_ratio = amount_concentration * amount_strength * amount_rank * price_amount_alignment

    volume_change = vol.pct_change()
    price_change_1 = close.pct_change()
    volume_shock = volume_change / volume_change.rolling(20).std().replace(0, np.nan)
    price_shock = price_change_1 / price_change_1.rolling(20).std().replace(0, np.nan)
    bb_width_abs = (upper_band - lower_band).abs()
    bb_width_ma = bb_width_abs.rolling(20).mean()
    vol_compression = np.exp(-bb_width_abs / bb_width_ma.replace(0, np.nan))
    adx = compute_adx(high, low, close, 14) / 100.0
    liquidity_shock = volume_shock * price_shock * vol_compression * adx

    volume_ma_short = volume_ma5.shift(1)
    volume_ma_long = volume_ma20.shift(1)
    volume_ratio_short = vol / volume_ma_short.replace(0, np.nan)
    volume_ratio_long = vol / volume_ma_long.replace(0, np.nan)
    volume_quantile = vol.rolling(20).apply(lambda x: pd.Series(x).quantile(0.75), raw=False).shift(1)
    volume_break = (vol > volume_quantile).astype(float)
    price_trend = np.sign(close.pct_change(3))
    volume_cluster = volume_ratio_short * volume_ratio_long * volume_break * price_trend
    volume_cluster = np.sign(volume_cluster) * np.log1p(np.abs(volume_cluster))

    kc_width = 2 * atr20 / ma20.replace(0, np.nan)
    squeeze = (kc_width - bb_width) / bb_width.rolling(20).std().replace(0, np.nan)
    price_position = (close - ma20) / atr14.replace(0, np.nan)
    rsi_factor = np.exp(-rsi14 / 100.0)
    price_squeeze = squeeze * price_position * rsi_factor

    typical_price = (high + low + close) / 3.0
    vwap = (typical_price * vol).rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)
    vwap_dev = (close - vwap) / close.rolling(20).std().replace(0, np.nan)
    days_above_vwap = (close > vwap).astype(int).rolling(5).sum()
    days_factor = np.where(days_above_vwap >= 3, 1.2, 1.0)
    vwap_deviation = vwap_dev * volume_ratio10 * np.sign(close - vwap) * days_factor

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_strength = macd / signal.replace(0, np.nan)
    rsi_strength = (rsi14 - 50) / 20.0
    ma_diff = (ma5 - ma20) / std20.replace(0, np.nan)
    volume_trend = volume_ma5 / volume_ma20.replace(0, np.nan)
    trend_consistency = (close > ma20).astype(int).rolling(10).sum() / 10.0
    trend_strength = ma_diff * rsi_strength * macd_strength * volume_trend * trend_consistency

    resistance = high.rolling(20).max().shift(1)
    price_range = high.rolling(20).max() - low.rolling(20).min()
    break_intensity = (close - resistance) / price_range.replace(0, np.nan)
    support_resistance_break = (close > resistance).astype(float) * break_intensity * volume_ratio10

    def bottom_volume_reversal(
        lookback_days: int = 20,
        confirmation_days: int = 3,
    ) -> pd.Series:
        signals = pd.Series(0.0, index=close.index)
        for i in range(lookback_days + confirmation_days, len(close)):
            price_decline_20 = close.iloc[i] / close.iloc[max(0, i - 20)] - 1
            price_decline_5 = close.iloc[i] / close.iloc[max(0, i - 5)] - 1
            is_downtrend = (price_decline_20 < -0.15) and (price_decline_5 < -0.05)
            if not is_downtrend:
                continue

            has_volume_breakout = False
            breakout_day_idx = -1
            for j in range(1, 6):
                check_idx = i - j
                if check_idx < 0:
                    break
                volume_condition = volume_ratio20.iloc[check_idx] > 2.0
                price_condition = (close.iloc[check_idx] / open_.iloc[check_idx] - 1) > 0.03
                is_positive = close.iloc[check_idx] > open_.iloc[check_idx]
                if volume_condition and price_condition and is_positive:
                    has_volume_breakout = True
                    breakout_day_idx = check_idx
                    break

            if not has_volume_breakout:
                continue

            if breakout_day_idx < i:
                days_since_breakout = i - breakout_day_idx
                if 2 <= days_since_breakout <= 5:
                    breakout_high = high.iloc[breakout_day_idx]
                    breakout_close = close.iloc[breakout_day_idx]
                    breakout_volume = vol.iloc[breakout_day_idx]

                    pullback_from_high = close.iloc[i] / breakout_high - 1
                    volume_contraction = vol.iloc[i] / breakout_volume if breakout_volume else 1.0

                    is_pullback_depth = -0.10 < pullback_from_high < -0.03
                    is_volume_contraction = volume_contraction < 0.7
                    above_breakout_close = close.iloc[i] > breakout_close * 0.98

                    no_big_negative = True
                    for k in range(breakout_day_idx + 1, i + 1):
                        is_negative = close.iloc[k] < open_.iloc[k]
                        decline = (close.iloc[k] - open_.iloc[k]) / open_.iloc[k]
                        is_big_decline = decline < -0.04
                        is_high_volume = volume_ratio20.iloc[k] > 1.5
                        if is_negative and is_big_decline and is_high_volume:
                            no_big_negative = False
                            break

                    if (
                        is_pullback_depth
                        and is_volume_contraction
                        and above_breakout_close
                        and no_big_negative
                    ):
                        signals.iloc[i] = 1.0

        return signals

    def consolidation_breakout_pullback(
        consolidation_days: int = 20,
        pullback_window: int = 5,
    ) -> pd.Series:
        signals = pd.Series(0.0, index=close.index)
        for i in range(consolidation_days + pullback_window, len(close)):
            start_idx = max(0, i - consolidation_days)
            consolidation_range = high.iloc[start_idx:i].max() - low.iloc[start_idx:i].min()
            consolidation_volatility = consolidation_range / close.iloc[start_idx]
            is_consolidation = consolidation_volatility < 0.08
            if not is_consolidation:
                continue

            consolidation_high = high.iloc[start_idx:i].max()
            has_breakout = False
            breakout_day_idx = -1
            for j in range(pullback_window, 0, -1):
                check_idx = i - j
                if check_idx < start_idx:
                    continue
                is_price_breakout = close.iloc[check_idx] > consolidation_high
                is_volume_breakout = volume_ratio20.iloc[check_idx] > 1.8
                if is_price_breakout and is_volume_breakout:
                    has_breakout = True
                    breakout_day_idx = check_idx
                    break

            if not has_breakout:
                continue

            if breakout_day_idx < i:
                days_since_breakout = i - breakout_day_idx
                if 2 <= days_since_breakout <= 5:
                    breakout_price = close.iloc[breakout_day_idx]
                    breakout_high = high.iloc[breakout_day_idx]
                    current_pullback = close.iloc[i] / breakout_high - 1
                    breakout_volume = vol.iloc[breakout_day_idx]
                    current_volume_ratio = vol.iloc[i] / breakout_volume if breakout_volume else 1.0

                    is_pullback_depth = -0.08 < current_pullback < -0.02
                    is_volume_contraction = current_volume_ratio < 0.6
                    above_breakout_price = close.iloc[i] > breakout_price * 0.98

                    no_destructive_candle = True
                    for k in range(breakout_day_idx + 1, i + 1):
                        if k == breakout_day_idx + 1:
                            is_big_negative = (close.iloc[k] / open_.iloc[k] - 1) < -0.05
                            if is_big_negative:
                                no_destructive_candle = False
                                break
                        is_negative = candle_body.iloc[k] < -0.04
                        is_high_volume = volume_ratio20.iloc[k] > 1.5
                        if is_negative and is_high_volume:
                            no_destructive_candle = False
                            break

                    if (
                        is_pullback_depth
                        and is_volume_contraction
                        and above_breakout_price
                        and no_destructive_candle
                    ):
                        signals.iloc[i] = 1.0

        return signals

    def n_pattern_pullback(wave_length: int = 10) -> pd.Series:
        signals = pd.Series(0.0, index=close.index)
        half_window = max(1, wave_length // 2)
        lookback = wave_length * 2

        for i in range(wave_length * 3, len(close)):
            l1_idx = -1
            l1_price = float("inf")
            for j in range(1, lookback + 1):
                check_idx = i - j
                if check_idx < 0:
                    continue
                left_bound = max(0, check_idx - half_window)
                right_bound = min(len(close) - 1, check_idx + half_window)
                is_valley = low.iloc[check_idx] == low.iloc[left_bound : right_bound + 1].min()
                if is_valley and low.iloc[check_idx] < l1_price:
                    l1_idx = check_idx
                    l1_price = low.iloc[check_idx]

            if l1_idx < 0:
                continue

            h1_idx = -1
            h1_price = 0.0
            for j in range(half_window, wave_length + 1):
                check_idx = l1_idx + j
                if check_idx >= i:
                    break
                left_bound = max(0, check_idx - half_window)
                right_bound = min(len(close) - 1, check_idx + half_window)
                is_peak = high.iloc[check_idx] == high.iloc[left_bound : right_bound + 1].max()
                if is_peak and high.iloc[check_idx] > h1_price:
                    h1_idx = check_idx
                    h1_price = high.iloc[check_idx]

            if h1_idx < 0 or h1_idx >= i:
                continue

            l2_idx = -1
            l2_price = float("inf")
            for j in range(half_window, wave_length + 1):
                check_idx = h1_idx + j
                if check_idx >= i:
                    break
                left_bound = max(0, check_idx - half_window)
                right_bound = min(len(close) - 1, check_idx + half_window)
                is_valley = low.iloc[check_idx] == low.iloc[left_bound : right_bound + 1].min()
                if is_valley and low.iloc[check_idx] < l2_price:
                    l2_idx = check_idx
                    l2_price = low.iloc[check_idx]

            if l2_idx < 0 or l2_price <= l1_price:
                continue

            is_n_pattern = l2_price > l1_price * 1.01
            if not is_n_pattern:
                continue

            rally_high_idx = -1
            rally_high_price = 0.0
            for j in range(1, wave_length + 1):
                check_idx = l2_idx + j
                if check_idx >= i:
                    break
                if high.iloc[check_idx] > rally_high_price:
                    rally_high_idx = check_idx
                    rally_high_price = high.iloc[check_idx]

            if rally_high_idx < 0 or rally_high_idx >= i:
                continue

            pullback_from_high = close.iloc[i] / rally_high_price - 1
            is_pullback = -0.08 < pullback_from_high < -0.03
            if not is_pullback:
                continue

            rally_volume_avg = vol.iloc[l2_idx : rally_high_idx + 1].mean()
            volume_contraction = vol.iloc[i] / rally_volume_avg if rally_volume_avg else 1.0
            is_volume_contraction = volume_contraction < 0.7
            if not is_volume_contraction:
                continue

            no_destructive_candle = True
            for j in range(rally_high_idx + 1, i + 1):
                is_big_negative = candle_body.iloc[j] < -0.04
                is_high_volume = volume_ratio20.iloc[j] > 1.5
                if is_big_negative and is_high_volume:
                    if j > 1 and close.iloc[j] < close.iloc[j - 2]:
                        no_destructive_candle = False
                        break

            if is_pullback and is_volume_contraction and no_destructive_candle:
                signals.iloc[i] = 1.0

        return signals

    def triple_pattern_capture() -> pd.Series:
        bottom_factor = bottom_volume_reversal()
        consolidation_factor = consolidation_breakout_pullback()
        n_pattern_factor = n_pattern_pullback()
        health_score = volume_price_health_score()
        pullback_score = pullback_depth_volume_match()

        combined = (
            0.25 * bottom_factor
            + 0.25 * consolidation_factor
            + 0.20 * n_pattern_factor
            + 0.15 * health_score
            + 0.15 * pullback_score
        )
        return combined.clip(lower=0.0, upper=1.0).fillna(0)

    def volume_price_health_score(window: int = 20) -> pd.Series:
        scores = pd.Series(0.0, index=close.index)
        for i in range(window, len(close)):
            start_idx = i - window + 1
            window_data = slice(start_idx, i + 1)
            returns_window = ret.iloc[window_data]
            volume_window = vol.iloc[window_data]
            close_window = close.iloc[window_data]
            high_window = high.iloc[window_data]
            low_window = low.iloc[window_data]

            up_days = returns_window[returns_window > 0]
            down_days = returns_window[returns_window < 0]

            if len(up_days) > 0 and len(down_days) > 0:
                avg_volume_up = volume_window[up_days.index].mean()
                avg_volume_down = volume_window[down_days.index].mean()
                volume_ratio = avg_volume_up / avg_volume_down if avg_volume_down else 2.0
            else:
                volume_ratio = 1.0
            volume_ratio = min(volume_ratio, 3.0)

            big_up_days = returns_window[returns_window > 0.03]
            big_down_days = returns_window[returns_window < -0.03]
            if len(big_up_days) > 0:
                big_up_volume_mean = volume_window[big_up_days.index].mean()
                window_volume_mean = volume_window.mean()
                big_up_ratio = big_up_volume_mean / window_volume_mean if window_volume_mean else 1.0
            else:
                big_up_ratio = 1.0
            if len(big_down_days) > 0:
                big_down_volume_mean = volume_window[big_down_days.index].mean()
                window_volume_mean = volume_window.mean()
                big_down_ratio = big_down_volume_mean / window_volume_mean if window_volume_mean else 1.0
            else:
                big_down_ratio = 1.0

            big_candle_score = min(big_up_ratio / max(big_down_ratio, 0.5), 3.0)

            window_low = low_window.min()
            window_high = high_window.max()
            price_range = window_high - window_low
            if price_range > 0:
                current_position = (close.iloc[i] - window_low) / price_range
                position_score = 1.0 if 0.3 <= current_position <= 0.7 else 0.5
            else:
                position_score = 0.5

            ma5_window = close_window.iloc[-5:].mean() if len(close_window) >= 5 else close.iloc[i]
            ma10_window = close_window.iloc[-10:].mean() if len(close_window) >= 10 else close.iloc[i]
            ma20_window = close_window.mean()
            if ma5_window > ma10_window > ma20_window:
                trend_score = 1.2
            elif ma5_window > ma10_window and ma10_window > ma20_window * 0.98:
                trend_score = 1.0
            else:
                trend_score = 0.8

            health_score = (
                0.25 * volume_ratio
                + 0.25 * big_candle_score
                + 0.20 * position_score
                + 0.20 * trend_score
                + 0.10 * (1.0 if volume_ratio > 1.2 else 0.8)
            )
            scores.iloc[i] = health_score

        min_score = scores.min()
        max_score = scores.max()
        if max_score > min_score:
            scores = (scores - min_score) / (max_score - min_score)
        return scores.fillna(0.5)

    def pullback_depth_volume_match(lookback_days: int = 10) -> pd.Series:
        scores = pd.Series(0.0, index=close.index)
        for i in range(lookback_days, len(close)):
            recent_high_idx = -1
            recent_high_price = 0.0
            for j in range(1, lookback_days + 1):
                check_idx = i - j
                if check_idx < 0:
                    break
                if high.iloc[check_idx] > recent_high_price:
                    recent_high_idx = check_idx
                    recent_high_price = high.iloc[check_idx]

            if recent_high_idx < 0:
                continue

            current_price = close.iloc[i]
            pullback_depth = current_price / recent_high_price - 1
            if not (-0.15 < pullback_depth < -0.02):
                continue

            rally_start_idx = max(0, recent_high_idx - 3)
            rally_volume_avg = vol.iloc[rally_start_idx : recent_high_idx + 1].mean()
            pullback_volume_avg = vol.iloc[recent_high_idx + 1 : i + 1].mean()
            if rally_volume_avg > 0:
                volume_contraction = pullback_volume_avg / rally_volume_avg
            else:
                volume_contraction = 1.0

            has_support = False
            for j in range(10, min(30, i)):
                support_level = low.iloc[i - j]
                if support_level and abs(current_price / support_level - 1) < 0.02:
                    has_support = True
                    break

            has_reversal_signal = False
            if i > 0:
                current_candle_body = candle_body.iloc[i]
                prev_candle_body = candle_body.iloc[i - 1]
                long_lower_shadow = lower_shadow.iloc[i] > abs(current_candle_body) * 2
                small_body = abs(current_candle_body) < 0.01
                if (
                    current_candle_body > 0
                    and prev_candle_body < 0
                    and close.iloc[i] > open_.iloc[i - 1]
                    and open_.iloc[i] < close.iloc[i - 1]
                ):
                    has_reversal_signal = True
                elif long_lower_shadow or small_body:
                    has_reversal_signal = True

            if -0.08 <= pullback_depth <= -0.03:
                depth_score = 1.0
            elif -0.12 <= pullback_depth < -0.08:
                depth_score = 0.8
            elif -0.03 < pullback_depth <= -0.02:
                depth_score = 0.7
            else:
                depth_score = 0.5

            if volume_contraction < 0.6:
                volume_score = 1.0
            elif volume_contraction < 0.8:
                volume_score = 0.8
            elif volume_contraction < 1.0:
                volume_score = 0.6
            else:
                volume_score = 0.4

            support_score = 1.0 if has_support else 0.6
            reversal_score = 1.0 if has_reversal_signal else 0.7
            above_ma20 = current_price > ma20.iloc[i] * 0.98 if not np.isnan(ma20.iloc[i]) else False
            position_score = 1.0 if above_ma20 else 0.5

            scores.iloc[i] = np.mean(
                [depth_score, volume_score, support_score, reversal_score, position_score]
            )

        return scores.fillna(0)

    bottom_factor = bottom_volume_reversal()
    consolidation_factor = consolidation_breakout_pullback()
    n_pattern_factor = n_pattern_pullback()
    triple_pattern = triple_pattern_capture()
    health_score = volume_price_health_score()
    pullback_match = pullback_depth_volume_match()

    latest = {
        "VAM50": vam50.iloc[-1],
        "REV50": rev50.iloc[-1],
        "LVOL50": lvol50.iloc[-1],
        "PVDC50": pvdc50.iloc[-1],
        "MSI50": msi50.iloc[-1],
        "BreakoutFollow_50": breakout_follow_50.iloc[-1],
        "DrawdownPressure_50": drawdown_pressure_50.iloc[-1],
        "VolatilityExpansion_50": volatility_expansion_50.iloc[-1],
        "PriceVolumeCoMove_50": price_volume_comove_50.iloc[-1],
        "UpperWickRatio_50": upper_wick_ratio_50.iloc[-1],
        "OpenCloseDominance_50": open_close_dominance_50.iloc[-1],
        "VolumeImpulse_50": volume_impulse_50.iloc[-1],
        "Momentum_Breakthrough_Combo": momentum_break.iloc[-1],
        "Volume_Contraction_Reversal": extreme_reversal.iloc[-1],
        "Volatility_Squeeze_Breakout": vol_regime.iloc[-1],
        "Amount_Concentration_Ratio": amount_concentration_ratio.iloc[-1],
        "Liquidity_Shock_Adaptive": liquidity_shock.iloc[-1],
        "Volume_Cluster_Ratio": volume_cluster.iloc[-1],
        "Price_Squeeze_Indicator": price_squeeze.iloc[-1],
        "VWAP_Deviation": vwap_deviation.iloc[-1],
        "Trend_Strength_Composite": trend_strength.iloc[-1],
        "Support_Resistance_Break": support_resistance_break.iloc[-1],
        "Bottom_Volume_Reversal": bottom_factor.iloc[-1],
        "Consolidation_Breakout_Pullback": consolidation_factor.iloc[-1],
        "N_Pattern_Pullback": n_pattern_factor.iloc[-1],
        "Triple_Pattern_Capture": triple_pattern.iloc[-1],
        "Volume_Price_Health_Score": health_score.iloc[-1],
        "Pullback_Depth_Volume_Match": pullback_match.iloc[-1],
    }
    return latest


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / std


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def infer_reference_codes(data_dir: str) -> List[str]:
    codes = []
    if not os.path.isdir(data_dir):
        return codes
    for name in os.listdir(data_dir):
        if not name.endswith(".csv"):
            continue
        if name.startswith("factor_") or name.startswith("B1_"):
            continue
        code = name.split("-")[0]
        if code and code not in codes:
            codes.append(code)
    return sorted(codes)


def main() -> None:
    parser = argparse.ArgumentParser(description="KDJ pool selection and factor scoring.")
    parser.add_argument("--pool-sector", default="", help="sector code/name for pool")
    parser.add_argument("--block-type", type=int, default=0)
    parser.add_argument("--pool-type", default="5", help="get_stock_list type code")
    parser.add_argument("--end-time", default="", help="YYYYMMDD or YYYYMMDDHHMMSS (blank = latest)")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--topn", type=int, default=10)
    base_dir = os.path.dirname(__file__)
    parser.add_argument("--out-dir", default=os.path.join(base_dir, "R_DATA"))
    parser.add_argument("--factor-set", choices=["5", "6", "7", "10"], default="7")
    parser.add_argument(
        "--factor-list",
        default="",
        help="comma-separated factor names to override factor-set",
    )
    parser.add_argument("--use-kdj-file", default="", help="path to cached KDJ CSV")
    parser.add_argument("--ref-codes", default="", help="comma-separated reference codes")
    parser.add_argument("--ref-file", default="", help="path to reference code list")
    parser.add_argument("--ref-dir", default=os.path.join(base_dir, "B1_DATA"))
    parser.add_argument(
        "--ref-mode",
        choices=["mean", "best", "median"],
        default="mean",
        help="reference similarity mode",
    )
    args = parser.parse_args()

    tq.initialize(__file__)

    if args.pool_type:
        pool_codes = tq.get_stock_list(args.pool_type)
    elif args.pool_sector:
        pool_codes = tq.get_stock_list_in_sector(args.pool_sector, block_type=args.block_type)
    else:
        raise ValueError("pool not specified. use --pool-type or --pool-sector")

    if not pool_codes:
        raise RuntimeError("empty pool from TDX")

    if args.factor_list:
        factor_names = [f.strip() for f in args.factor_list.split(",") if f.strip()]
        valid = set(ALL_FACTORS)
        unknown = [f for f in factor_names if f not in valid]
        if unknown:
            raise ValueError(f"unknown factors: {unknown}")
    else:
        if args.factor_set == "5":
            factor_names = FACTOR_SET_5
        elif args.factor_set == "6":
            factor_names = FACTOR_SET_6
        elif args.factor_set == "10":
            factor_names = FACTOR_SET_10
        else:
            factor_names = FACTOR_SET_7

    if args.ref_codes:
        ref_codes = [c.strip() for c in args.ref_codes.split(",") if c.strip()]
    elif args.ref_file:
        with open(args.ref_file, "r", encoding="utf-8") as f:
            ref_codes = [line.strip() for line in f if line.strip()]
    else:
        ref_codes = infer_reference_codes(args.ref_dir)
    if not ref_codes:
        raise RuntimeError("reference codes not provided or not found")

    os.makedirs(args.out_dir, exist_ok=True)
    kdj_out = os.path.join(args.out_dir, "B1_kdj_candidates.csv")

    if args.use_kdj_file:
        kdj_df = pd.read_csv(args.use_kdj_file)
        if "Code" not in kdj_df.columns:
            raise ValueError("use_kdj_file must include Code column")
        if "J" in kdj_df.columns:
            kdj_candidates = kdj_df[kdj_df["J"] < 13].copy()
        else:
            raise ValueError("use_kdj_file must include J column for filtering")
        candidate_codes = kdj_candidates["Code"].dropna().tolist()
        pool_codes = sorted(set(candidate_codes).union(ref_codes))
        ohlcv = fetch_ohlcv(pool_codes, args.count, args.end_time)
    else:
        ohlcv = fetch_ohlcv(pool_codes, args.count, args.end_time)
        close_df = ohlcv["Close"]
        high_df = ohlcv["High"]
        low_df = ohlcv["Low"]
        open_df = ohlcv["Open"]
        vol_df = ohlcv["Volume"]

        latest_date = close_df.index[-1]
        kdj_rows = []
        for code in close_df.columns:
            j = compute_kdj_j(high_df[code], low_df[code], close_df[code])
            j_last = j.iloc[-1]
            kdj_rows.append({"Code": code, "Date": latest_date, "J": j_last})
        kdj_df = pd.DataFrame(kdj_rows).sort_values("J")
        kdj_candidates = kdj_df[kdj_df["J"] < 13].copy()
        kdj_df.to_csv(kdj_out, index=False, encoding="utf-8-sig")

        if kdj_candidates.empty:
            print("no KDJ candidates with J < 13")
            return

    close_df = ohlcv["Close"]
    high_df = ohlcv["High"]
    low_df = ohlcv["Low"]
    open_df = ohlcv["Open"]
    vol_df = ohlcv["Volume"]

    ref_set = set(ref_codes)
    pool_set = set(close_df.columns)
    ref_codes = [c for c in ref_codes if c in pool_set]
    if not ref_codes:
        raise RuntimeError("reference codes not in pool")

    if not args.use_kdj_file:
        candidate_codes = kdj_candidates["Code"].tolist()
    else:
        candidate_codes = kdj_candidates["Code"].dropna().tolist()
    if not candidate_codes:
        print("no KDJ candidates with J < 13")
        return
    union_codes = sorted(set(candidate_codes).union(ref_codes))

    factor_rows = []
    for code in union_codes:
        latest = compute_factors_for_stock(
            open_df[code],
            high_df[code],
            low_df[code],
            close_df[code],
            vol_df[code],
        )
        row = {"Code": code}
        row.update(latest)
        factor_rows.append(row)

    factor_df = pd.DataFrame(factor_rows)
    z_df = factor_df.copy()
    for factor in factor_names:
        z_df[f"{factor}_z"] = zscore(z_df[factor])

    ref_mask = z_df["Code"].isin(ref_codes)
    ref_vectors = z_df.loc[ref_mask, [f"{f}_z" for f in factor_names]].to_numpy()
    if ref_vectors.size == 0:
        raise RuntimeError("reference vectors are empty after zscore")
    # Drop columns that are all NaN across references to avoid unstable similarity.
    col_mask = ~np.all(np.isnan(ref_vectors), axis=0)
    ref_vectors = ref_vectors[:, col_mask]
    if ref_vectors.size == 0:
        raise RuntimeError("reference vectors are all NaN after zscore")
    if args.ref_mode == "median":
        ref_center = np.nanmedian(ref_vectors, axis=0)
    else:
        ref_center = np.nanmean(ref_vectors, axis=0)
    if np.isnan(ref_center).all():
        raise RuntimeError("reference vectors are all NaN after zscore")
    # Fill remaining NaNs with reference center for robust best-mode similarity.
    ref_vectors = np.where(np.isnan(ref_vectors), ref_center, ref_vectors)

    candidate_mask = z_df["Code"].isin(candidate_codes)
    sim_rows = []
    for _, row in z_df.loc[candidate_mask].iterrows():
        vec = row[[f"{f}_z" for f in factor_names]].to_numpy(dtype=float)
        vec = vec[col_mask]
        vec = np.where(np.isnan(vec), ref_center, vec)
        if args.ref_mode == "best":
            sims = []
            for ref_vec in ref_vectors:
                mask = ~np.isnan(ref_vec)
                if mask.sum() < 5:
                    continue
                sims.append(cosine_similarity(vec[mask], ref_vec[mask]))
            if not sims:
                continue
            sim = float(np.nanmax(sims))
        else:
            mask = ~np.isnan(vec) & ~np.isnan(ref_center)
            if mask.sum() < 5:
                continue
            sim = cosine_similarity(vec[mask], ref_center[mask])
        sim_rows.append({"Code": row["Code"], "Similarity": sim})

    sim_df = pd.DataFrame(sim_rows).sort_values("Similarity", ascending=False)
    top_df = sim_df.head(args.topn)

    suffix = "_custom" if args.factor_list else f"_{args.factor_set}f"
    z_out = os.path.join(args.out_dir, f"B1_factor_zscores{suffix}.csv")
    sim_out = os.path.join(args.out_dir, f"B1_similarity{suffix}.csv")
    top_out = os.path.join(args.out_dir, f"B1_top10{suffix}.csv")
    z_df.to_csv(z_out, index=False, encoding="utf-8-sig")
    sim_df.to_csv(sim_out, index=False, encoding="utf-8-sig")
    top_df.to_csv(top_out, index=False, encoding="utf-8-sig")

    print("saved:", kdj_out)
    print("saved:", z_out)
    print("saved:", sim_out)
    print("saved:", top_out)


if __name__ == "__main__":
    main()
