# ZGNB

Quant workflow scripts for TDX data, factor research, and B1 selection.

## Contents

- tdxdata_download.py: Download forward-adjusted daily bars.
- factor_batch.py: Batch factor computation and summaries.
- factor_similarity.py: Similarity scoring for factor panels.
- workflow_B1.py: KDJ filter + factor z-score + similarity + Top10.
- backtest_B1_7f_nextday.py: Next-day backtest for factor set 7.

## Folders

- B1_DATA: Reference data (ignored by git).
- R_DATA: Generated outputs (ignored by git).

## Quick Start

1. Put reference CSVs into B1_DATA.
2. Run the B1 workflow:

```bash
python workflow_B1.py --factor-set 6
```

Outputs are written to R_DATA.

## Factor Sets

### F5 (Base)

- VAM50
	- Math: $F=Z_{50}(V)\cdot Z_{50}(\sum_{i=1}^{10} r_i)$
	- Logic: Volume and short-term returns move together, supporting momentum.
- REV50
	- Math: $F=-Z_{50}(\sum_{i=1}^{20} r_i)$
	- Logic: Large medium-term gains imply mean-reversion pressure.
- LVOL50
	- Math: $F=-\sigma_{50}(r)$
	- Logic: Lower volatility signals a more stable profile.
- PVDC50
	- Math: $F=-\mathrm{Corr}_{50}(\Delta P,\Delta V)$
	- Logic: Weak price-volume co-movement suggests re-pricing potential.
- MSI50
	- Math: $F=Z_{50}(((H-L)/C)\cdot V)$
	- Logic: Range and volume expansion indicates activity strength.

### F7 (Structure)

- BreakoutFollow_50
	- Math: $F=\frac{C-H^{(50)}_{t-1}}{H^{(50)}_{t-1}}\cdot\frac{V}{MA_{50}(V)}$
	- Logic: Breakout strength with volume confirmation.
- DrawdownPressure_50
	- Math: $F=\frac{\max_{50}(C)-C}{\max_{50}(C)}$
	- Logic: Larger drawdowns imply stronger mean-reversion pressure.
- VolatilityExpansion_50
	- Math: $F=\Delta\sigma_{50}(r)$
	- Logic: Rising volatility often accompanies trend shifts.
- PriceVolumeCoMove_50
	- Math: $F=\mathrm{Corr}_{50}(\Delta C,\Delta V)$
	- Logic: Price and volume moving together suggests trend persistence.
- UpperWickRatio_50
	- Math: $F=MA_{50}((H-\max(C,O))/(H-L))$
	- Logic: Larger upper wicks indicate sell pressure.
- OpenCloseDominance_50
	- Math: $F=MA_{50}((C-O)/(H-L))$
	- Logic: Close relative to open reflects directional control.
- VolumeImpulse_50
	- Math: $F=V/MA_{50}(V)$
	- Logic: Volume bursts highlight participation surges.

### F10 (Composite)

- Momentum_Breakthrough_Combo
	- Math: $F=\frac{V}{MA_{10}(V)}\cdot\frac{C}{H^{(5)}_{t-1}}\cdot\frac{RSI_6}{50}\cdot\mathrm{sign}(C-MA_{20})$
	- Logic: Momentum plus breakout plus RSI alignment.
- Volume_Contraction_Reversal
	- Math: $F=(1-\frac{V}{MA_5(V)})\cdot\frac{C-L^{(5)}}{ATR_{14}}\cdot\mathrm{sign}(MA_5-MA_{20})$
	- Logic: Shrinking volume followed by price rebound.
- Volatility_Squeeze_Breakout
	- Math: $F=\frac{BB\_width}{ATR/C}\cdot\frac{C-MA_{20}}{\sigma_{20}}\cdot\mathrm{sign}(V-MA_{10})$
	- Logic: Volatility squeeze with directional release.
- Amount_Concentration_Ratio
	- Math: $F=\frac{Amt}{\sum_5 Amt}\cdot\frac{Amt}{MA_{20}(Amt)}\cdot\mathrm{Rank}_{60}(Amt)\cdot\mathrm{sign}(\Delta P)\mathrm{sign}(\Delta Amt)$
	- Logic: Concentrated turnover with aligned price move.
- Liquidity_Shock_Adaptive
	- Math: $F=\frac{\Delta V}{\sigma_{20}(\Delta V)}\cdot\frac{\Delta P}{\sigma_{20}(\Delta P)}\cdot e^{-BB\_width/MA(BB\_width)}\cdot ADX_{14}$
	- Logic: Liquidity shocks confirmed by trend strength.
- Volume_Cluster_Ratio
	- Math: $F=\log(1+|\frac{V}{MA_5}\cdot\frac{V}{MA_{20}}\cdot I(V>Q_{0.75})\cdot\mathrm{sign}(\Delta P)|)$
	- Logic: Volume clustering with aligned price movement.
- Price_Squeeze_Indicator
	- Math: $F=\frac{KC\_width-BB\_width}{\sigma_{20}(BB\_width)}\cdot\frac{C-MA_{20}}{ATR_{14}}\cdot e^{-RSI_{14}/100}$
	- Logic: Squeeze state with position and RSI filter.
- VWAP_Deviation
	- Math: $F=\frac{C-VWAP_{20}}{\sigma_{20}(C)}\cdot\frac{V}{MA_{10}(V)}\cdot\mathrm{sign}(C-VWAP)\cdot f(\text{above VWAP})$
	- Logic: Directional deviation from VWAP with volume support.
- Trend_Strength_Composite
	- Math: $F=\frac{MA_5-MA_{20}}{\sigma_{20}}\cdot\frac{RSI_{14}-50}{20}\cdot\frac{MACD}{Signal}\cdot\frac{MA_5(V)}{MA_{20}(V)}\cdot\mathrm{Consistency}$
	- Logic: Multi-indicator trend strength score.
- Support_Resistance_Break
	- Math: $F=I(C>R)\cdot\frac{C-R}{R-L}\cdot\frac{V}{MA_{10}(V)}$
	- Logic: Break above resistance with volume confirmation.

### F6 (Real-Time Patterns)

- Bottom_Volume_Reversal
	- Math: $F=I(\text{downtrend})\cdot I(\text{volume breakout})\cdot I(\text{pullback depth})\cdot I(\text{volume contraction})\cdot I(\text{no big negative})$
	- Logic: Completed bottom reversal with volume breakout and healthy pullback.
- Consolidation_Breakout_Pullback
	- Math: $F=I(\text{low volatility range})\cdot I(\text{breakout with volume})\cdot I(\text{pullback depth})\cdot I(\text{volume contraction})\cdot I(\text{no destructive candle})$
	- Logic: Box-range breakout followed by a controlled pullback.
- N_Pattern_Pullback
	- Math: $F=I(L_2>L_1)\cdot I(\text{rally after }L_2)\cdot I(\text{pullback }3\%-8\%)\cdot I(\text{volume contraction})\cdot I(\text{no destructive candle})$
	- Logic: Higher-high/higher-low structure with a healthy pullback.
- Triple_Pattern_Capture
	- Math: $F=0.25F_{bottom}+0.25F_{consolidation}+0.20F_{N}+0.15F_{health}+0.15F_{pullback}$
	- Logic: Weighted blend of three patterns plus health and pullback quality.
- Volume_Price_Health_Score
	- Math: $F=\mathrm{Norm}(0.25\cdot\frac{\bar V_{up}}{\bar V_{down}}+0.25\cdot\frac{V_{big\ up}}{V_{big\ down}}+0.20\cdot\text{Position}+0.20\cdot\text{Trend}+0.10\cdot\text{Volume Strength})$
	- Logic: Healthy structure favors rising volume on up days and stable positioning.
- Pullback_Depth_Volume_Match
	- Math: $F=\mathrm{Mean}(\text{depth score},\text{volume score},\text{support score},\text{reversal score},\text{position score})$
	- Logic: Pullback quality score using depth, volume, support, and candle signals.
