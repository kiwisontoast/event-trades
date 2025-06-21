import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class FilteredArbitrageTrader:
    """
    Real-time arbitrage trader for forecast percentage fluctuations.
    UPDATED for data BEFORE June 19, 2025 7pm (excludes major event changes)

    Strategy: Profit from mean reversion to baseline (69.0%)
    - When forecast drops below baseline: BUY (expect reversion up)
    - When forecast rises above baseline: SELL (expect reversion down)
    """

    def __init__(self):
        # Updated parameters based on filtered data analysis (before June 19 7pm)
        self.BASELINE = 69.0  # Most frequent value in filtered data (21.3% of time)
        self.MIN_DEVIATION = (
            0.5  # Minimum 0.5% deviation to trade (adjusted for filtered data)
        )
        self.TRANSACTION_COST = 0.002  # 0.2% transaction cost
        self.RISK_PER_TRADE = 0.02  # Risk 2% of capital per trade

        # Reversion probabilities based on filtered data patterns
        self.REVERSION_PROB_DOWN = 0.75  # 75% chance when below baseline
        self.REVERSION_PROB_UP = (
            0.75  # 75% chance when above baseline (more balanced in filtered data)
        )
        self.BASELINE_FREQUENCY = 0.213  # 21.3% of time at baseline in filtered data

    def analyze_current_opportunity(self, current_forecast):
        """
        Analyze current trading opportunity using filtered data parameters.

        Args:
            current_forecast (float): Current forecast percentage (e.g., 68.5)

        Returns:
            dict: Complete trading analysis
        """
        deviation = current_forecast - self.BASELINE
        abs_deviation = abs(deviation)

        # Base analysis
        analysis = {
            "current_forecast": current_forecast,
            "baseline": self.BASELINE,
            "deviation": deviation,
            "abs_deviation": abs_deviation,
            "deviation_percent": (abs_deviation / self.BASELINE) * 100,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Trading decision logic
        if abs_deviation < self.MIN_DEVIATION:
            analysis.update(
                {
                    "signal": "HOLD",
                    "confidence": 0,
                    "reason": f"Deviation {abs_deviation:.3f}% too small (min: {self.MIN_DEVIATION}%)",
                    "expected_profit": 0,
                    "risk_level": "NONE",
                }
            )

        elif deviation < 0:  # Below baseline - BUY signal
            confidence = self.REVERSION_PROB_DOWN
            # Increase confidence for larger deviations
            confidence_adj = min(0.95, confidence + (abs_deviation / 5.0) * 0.1)
            expected_move = abs_deviation * confidence_adj

            analysis.update(
                {
                    "signal": "BUY",
                    "confidence": confidence_adj,
                    "reason": f"Forecast {abs_deviation:.3f}% below baseline - expect upward reversion",
                    "expected_move": expected_move,
                    "expected_profit": expected_move * 0.8,  # Conservative estimate
                    "risk_level": self._calculate_risk_level(abs_deviation),
                    "stop_loss": current_forecast
                    - (abs_deviation * 0.4),  # Tighter stop for filtered data
                    "take_profit": self.BASELINE - 0.3,  # Target near baseline
                }
            )

        else:  # Above baseline - SELL signal
            confidence = self.REVERSION_PROB_UP
            # Increase confidence for larger deviations
            confidence_adj = min(0.95, confidence + (abs_deviation / 5.0) * 0.1)
            expected_move = abs_deviation * confidence_adj

            analysis.update(
                {
                    "signal": "SELL",
                    "confidence": confidence_adj,
                    "reason": f"Forecast {abs_deviation:.3f}% above baseline - expect downward reversion",
                    "expected_move": expected_move,
                    "expected_profit": expected_move * 0.8,
                    "risk_level": self._calculate_risk_level(abs_deviation),
                    "stop_loss": current_forecast
                    + (abs_deviation * 0.4),  # Tighter stop for filtered data
                    "take_profit": self.BASELINE + 0.3,  # Target near baseline
                }
            )

        return analysis

    def calculate_position_size(self, analysis, account_balance):
        """
        Calculate optimal position size using Kelly Criterion.

        Args:
            analysis (dict): Analysis from analyze_current_opportunity()
            account_balance (float): Current account balance

        Returns:
            dict: Position sizing recommendation
        """
        if analysis["signal"] == "HOLD":
            return {"position_size": 0, "risk_amount": 0}

        # Kelly Criterion with adjustments for filtered data
        win_prob = analysis["confidence"]
        lose_prob = 1 - win_prob
        profit_ratio = analysis["expected_profit"] / analysis["abs_deviation"]

        # Kelly fraction
        kelly_fraction = (profit_ratio * win_prob - lose_prob) / profit_ratio

        # Conservative sizing (use 30% of Kelly for filtered data - less volatile)
        conservative_fraction = max(0, min(kelly_fraction * 0.30, self.RISK_PER_TRADE))

        # Calculate position size
        risk_amount = account_balance * conservative_fraction
        position_size = risk_amount / (analysis["abs_deviation"] / 100)

        return {
            "position_size": position_size,
            "risk_amount": risk_amount,
            "kelly_fraction": kelly_fraction,
            "conservative_fraction": conservative_fraction,
            "max_loss": risk_amount,
        }

    def _calculate_risk_level(self, deviation):
        """Calculate risk level based on deviation magnitude for filtered data."""
        if deviation < 1.0:
            return "LOW"
        elif deviation < 2.0:
            return "MEDIUM"
        elif deviation < 4.0:
            return "HIGH"
        else:
            return "VERY HIGH"

    def get_trading_recommendation(self, current_forecast, account_balance=10000):
        """
        Get complete trading recommendation for current forecast.

        Args:
            current_forecast (float): Current forecast percentage
            account_balance (float): Available trading capital

        Returns:
            dict: Complete trading recommendation
        """
        # Analyze opportunity
        analysis = self.analyze_current_opportunity(current_forecast)

        # Calculate position size
        position_info = self.calculate_position_size(analysis, account_balance)

        # Combine analysis and position sizing
        recommendation = {**analysis, **position_info}

        # Add execution details
        if recommendation["signal"] != "HOLD":
            total_cost = recommendation["position_size"] * self.TRANSACTION_COST
            net_position = recommendation["position_size"] - total_cost

            recommendation.update(
                {
                    "transaction_cost": total_cost,
                    "net_position": net_position,
                    "cost_percentage": (
                        (total_cost / recommendation["position_size"]) * 100
                        if recommendation["position_size"] > 0
                        else 0
                    ),
                }
            )

        return recommendation

    def print_recommendation(self, recommendation):
        """Print formatted trading recommendation."""
        print("=" * 70)
        print(
            f"🎯 FILTERED DATA ARBITRAGE RECOMMENDATION - {recommendation['timestamp']}"
        )
        print(f"   (Based on data BEFORE June 19, 2025 7pm)")
        print("=" * 70)
        print(f"Current Forecast: {recommendation['current_forecast']:.3f}%")
        print(f"Baseline: {recommendation['baseline']:.3f}%")
        print(
            f"Deviation: {recommendation['deviation']:+.3f}% ({recommendation['deviation_percent']:.2f}%)"
        )
        print(f"Risk Level: {recommendation['risk_level']}")
        print()

        print(f"📊 SIGNAL: {recommendation['signal']}")
        print(f"Confidence: {recommendation['confidence']*100:.0f}%")
        print(f"Reasoning: {recommendation['reason']}")
        print()

        if recommendation["signal"] != "HOLD":
            print("💰 POSITION DETAILS:")
            print(f"Position Size: ${recommendation['position_size']:,.2f}")
            print(f"Risk Amount: ${recommendation['risk_amount']:,.2f}")
            print(f"Max Loss: ${recommendation['max_loss']:,.2f}")
            print(f"Expected Profit: {recommendation['expected_profit']:.3f}%")
            print(f"Transaction Cost: ${recommendation['transaction_cost']:,.2f}")
            print()

            print("🎯 TRADE LEVELS:")
            print(f"Take Profit: {recommendation['take_profit']:.3f}%")
            print(f"Stop Loss: {recommendation['stop_loss']:.3f}%")

        print("=" * 70)

    def analyze_data_before_cutoff(
        self, csv_file, cutoff_datetime="2025-06-19 19:00:00"
    ):
        """
        Analyze only the data before the specified cutoff time.

        Args:
            csv_file (str): Path to CSV file
            cutoff_datetime (str): Cutoff datetime string

        Returns:
            dict: Analysis results
        """
        df = pd.read_csv(csv_file)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        cutoff = pd.to_datetime(cutoff_datetime).tz_localize("UTC")

        # Filter data before cutoff
        filtered_df = df[df["Timestamp"] < cutoff].copy()
        filtered_df["Forecast_Value"] = (
            filtered_df["Forecast"].str.rstrip("%").astype(float)
        )
        # filtered_df = filtered_df[["Timestamp", "Forecast_Value"]]

        # Analyze baseline
        forecast_counts = filtered_df["Forecast_Value"].value_counts()
        baseline = forecast_counts.index[0]
        baseline_freq = forecast_counts.iloc[0] / len(filtered_df)

        # Analyze deviations
        deviations = filtered_df[filtered_df["Forecast_Value"] != baseline]
        avg_deviation = (
            abs(deviations["Forecast_Value"] - baseline).mean()
            if len(deviations) > 0
            else 0
        )

        print(f"📊 FILTERED DATA ANALYSIS (Before {cutoff_datetime})")
        print("=" * 60)
        print(f"Total records: {len(df):,}")
        print(
            f"Filtered records: {len(filtered_df):,} ({len(filtered_df)/len(df)*100:.1f}%)"
        )
        print(f"Baseline: {baseline:.1f}% ({baseline_freq*100:.1f}% of time)")
        print(f"Average deviation: {avg_deviation:.3f}%")
        print(f"Total deviations: {len(deviations):,}")
        timestamps = filtered_df["Timestamp"].tolist()

        return {
            "baseline": baseline,
            "baseline_frequency": baseline_freq,
            "avg_deviation": avg_deviation,
            "total_records": len(filtered_df),
            "deviation_count": len(deviations),
            "timestamps": timestamps,
            "forecasts": filtered_df["Forecast"].tolist(),
        }

        return {
            "baseline": baseline,
            "baseline_frequency": baseline_freq,
            "avg_deviation": avg_deviation,
            "total_records": len(filtered_df),
            "deviation_count": len(deviations),
        }

    def simulate_scenario(self, forecast_sequence, account_balance=10000):
        """
        Simulate trading across a sequence of forecasts.

        Args:
            forecast_sequence (list): List of forecast percentages
            account_balance (float): Starting balance

        Returns:
            dict: Simulation results
        """
        balance = account_balance
        trades = []
        total_profit = 0

        print(f"\n🔄 FILTERED DATA TRADING SIMULATION")
        print(f"Starting Balance: ${balance:,.2f}")
        print("-" * 50)

        for i, forecast in enumerate(forecast_sequence):
            recommendation = self.get_trading_recommendation(forecast, balance)

            if recommendation["signal"] != "HOLD":
                # Simulate trade execution
                profit_loss = (
                    recommendation["expected_profit"]
                    * recommendation["position_size"]
                    / 100
                )
                profit_loss -= recommendation["transaction_cost"]  # Subtract costs

                balance += profit_loss
                total_profit += profit_loss

                trade_record = {
                    "step": i + 1,
                    "forecast": forecast,
                    "signal": recommendation["signal"],
                    "position_size": recommendation["position_size"],
                    "profit_loss": profit_loss,
                    "balance": balance,
                }
                trades.append(trade_record)

                print(
                    f"Step {i+1}: {forecast:.2f}% -> {recommendation['signal']} -> "
                    f"P&L: ${profit_loss:+.2f} -> Balance: ${balance:,.2f}"
                )

        print("-" * 50)
        print(f"Final Balance: ${balance:,.2f}")
        print(f"Total Profit: ${total_profit:+,.2f}")
        print(f"Return: {((balance - account_balance) / account_balance * 100):+.2f}%")

        return {
            "final_balance": balance,
            "total_profit": total_profit,
            "return_percent": ((balance - account_balance) / account_balance * 100),
            "num_trades": len(trades),
            "trades": trades,
        }


# Usage Examples for Filtered Data
def main():
    """Demonstrate the arbitrage trading algorithm with filtered data."""
    trader = FilteredArbitrageTrader()

    # Example 1: Analyze the actual filtered dataset
    print("STEP 1: Analyzing filtered dataset")
    try:
        analysis = trader.analyze_data_before_cutoff("oklahomavsindianafinal.csv")
    except FileNotFoundError:
        print("CSV file not found - using simulated analysis")

    print("\n" + "=" * 70)

    # Example 2: Test various forecast scenarios based on filtered data
    print("STEP 2: Testing forecast scenarios (filtered data baseline: 69%)")
    test_forecasts = [
        67.0,  # 2% below baseline - moderate buy
        66.0,  # 3% below baseline - strong buy
        69.0,  # At baseline - hold
        70.0,  # 1% above baseline - moderate sell
        71.0,  # 2% above baseline - strong sell
        65.0,  # 4% below baseline - very strong buy
        72.0,  # 3% above baseline - very strong sell
    ]

    for forecast in test_forecasts:
        recommendation = trader.get_trading_recommendation(forecast)
        trader.print_recommendation(recommendation)
        print()

    # Example 3: Simulation with realistic forecast sequence
    print("STEP 3: Trading sequence simulation (filtered data patterns)")
    realistic_sequence = [
        69.0,
        68.0,
        67.0,
        68.5,
        69.0,
        70.0,
        71.0,
        69.5,
        69.0,
        66.0,
        68.0,
        69.0,
    ]
    results = trader.simulate_scenario(realistic_sequence)

    print(f"\n📈 SUMMARY FOR FILTERED DATA STRATEGY:")
    print(f"- Baseline: 69.0% (stable before event)")
    print(f"- Strategy: Mean reversion with 75% confidence")
    print(f"- Risk management: Tighter stops, conservative sizing")
    print(f"- Best for: Pre-event stable periods")


if __name__ == "__main__":
    main()
    main()
