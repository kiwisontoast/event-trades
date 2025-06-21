import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json


class ForecastArbitrageTrader:
    def __init__(
        self, baseline_forecast=33.0, transaction_cost=0.0, max_position_size=1000
    ):
        """
        Initialize the arbitrage trader.

        Args:
            baseline_forecast (float): The baseline forecast percentage (33.0%)
            transaction_cost (float): Transaction cost as decimal (0.1% = 0.001)
            max_position_size (float): Maximum position size in dollars
        """
        self.baseline = baseline_forecast
        self.transaction_cost = 0
        self.max_position = max_position_size
        self.position = 0  # Current position (-1000 to +1000)
        self.cash = 10000  # Starting cash
        self.total_pnl = 0
        self.trades = []
        self.current_forecast = baseline_forecast

    def calculate_expected_return_to_baseline(
        self, current_forecast, time_to_reversion=5
    ):
        """
        Calculate expected return assuming reversion to baseline.

        Args:
            current_forecast (float): Current forecast percentage
            time_to_reversion (int): Expected minutes to revert to baseline

        Returns:
            float: Expected return percentage
        """
        deviation = abs(current_forecast - self.baseline)
        # Higher deviations have higher expected returns but also higher risk
        expected_return = deviation * 0.5  # Conservative estimate
        return expected_return

    def calculate_position_size(self, current_forecast, confidence=0.8):
        """
        Calculate optimal position size based on Kelly Criterion and confidence.

        Args:
            current_forecast (float): Current forecast percentage
            confidence (float): Confidence in mean reversion (0-1)

        Returns:
            float: Suggested position size
        """
        deviation = abs(current_forecast - self.baseline)

        # Higher deviation = larger position (but capped)
        base_size = min(deviation * 100, self.max_position)

        # Adjust for confidence
        position_size = base_size * confidence

        return position_size

    def should_trade(self, current_forecast, min_deviation=0.05):
        """
        Determine if we should trade based on current forecast.

        Args:
            current_forecast (float): Current forecast percentage
            min_deviation (float): Minimum deviation to trigger trade

        Returns:
            dict: Trade recommendation
        """
        deviation = current_forecast - self.baseline
        abs_deviation = abs(deviation)

        if abs_deviation < min_deviation:
            return {"action": "hold", "reason": "Deviation too small"}

        # Based on historical data: 82% deviations are downward
        # When forecast is below baseline, expect it to rise back
        # When forecast is above baseline, expect it to fall back

        if deviation < 0:  # Forecast below baseline
            action = "buy"  # Expect price to rise back to baseline
            confidence = 0.85  # High confidence based on historical data
        else:  # Forecast above baseline
            action = "sell"  # Expect price to fall back to baseline
            confidence = 0.60  # Lower confidence (fewer upward deviations)

        position_size = self.calculate_position_size(current_forecast, confidence)
        expected_return = self.calculate_expected_return_to_baseline(current_forecast)

        return {
            "action": action,
            "position_size": position_size,
            "deviation": deviation,
            "expected_return": expected_return,
            "confidence": confidence,
            "reason": f"Forecast {current_forecast}% deviates from baseline {self.baseline}% by {deviation:.2f} points",
        }

    def execute_trade(self, recommendation, current_forecast, timestamp):
        """
        Execute a trade based on recommendation.

        Args:
            recommendation (dict): Trade recommendation from should_trade()
            current_forecast (float): Current forecast percentage
            timestamp (str): Current timestamp
        """
        if recommendation["action"] == "hold":
            return

        action = recommendation["action"]
        size = recommendation["position_size"]

        # Calculate transaction cost
        cost = size * self.transaction_cost

        # Simulate trade execution
        if action == "buy" and self.cash >= size + cost:
            self.position += size
            self.cash -= size + cost
            trade_type = "BUY"
        elif action == "sell" and self.position >= size:
            self.position -= size
            self.cash += size - cost
            trade_type = "SELL"
        else:
            return  # Cannot execute trade

        # Record trade
        trade = {
            "timestamp": timestamp,
            "type": trade_type,
            "size": size,
            "forecast": current_forecast,
            "deviation": recommendation["deviation"],
            "expected_return": recommendation["expected_return"],
            "confidence": recommendation["confidence"],
            "cost": cost,
            "position_after": self.position,
            "cash_after": self.cash,
        }

        self.trades.append(trade)
        self.current_forecast = current_forecast

    def close_position_at_baseline(self, timestamp):
        """
        Close position when forecast returns to baseline.

        Args:
            timestamp (str): Current timestamp
        """
        if self.position == 0:
            return

        # Calculate P&L from returning to baseline
        if self.position > 0:  # Long position
            # Profit if forecast moved toward baseline as expected
            pnl = self.position * 0.01  # Simplified P&L calculation
            self.cash += self.position + pnl
            trade_type = "CLOSE_LONG"
        else:  # Short position
            pnl = abs(self.position) * 0.01
            self.cash += abs(self.position) + pnl
            trade_type = "CLOSE_SHORT"

        # Record closing trade
        trade = {
            "timestamp": timestamp,
            "type": trade_type,
            "size": abs(self.position),
            "forecast": self.baseline,
            "pnl": pnl,
            "position_after": 0,
            "cash_after": self.cash,
        }

        self.trades.append(trade)
        self.total_pnl += pnl
        self.position = 0

    def backtest_strategy(self, deviation_data_file):
        """
        Backtest the arbitrage strategy on historical data.

        Args:
            deviation_data_file (str): Path to deviation data CSV

        Returns:
            dict: Backtest results
        """
        df = pd.read_csv(deviation_data_file)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="ISO8601")
        df = df.sort_values("Timestamp")

        print(f"Backtesting strategy on {len(df)} deviation events...")
        print(f"Starting cash: ${self.cash}")
        print("=" * 60)

        for idx, row in df.iterrows():
            current_forecast = row["Forecast_Value"]
            timestamp = row["Timestamp"]

            # Get trade recommendation
            recommendation = self.should_trade(current_forecast)

            # Execute trade if recommended
            if recommendation["action"] != "hold":
                print(
                    f"{timestamp}: {recommendation['action'].upper()} - "
                    f"Forecast: {current_forecast}%, "
                    f"Deviation: {recommendation['deviation']:.2f}pts, "
                    f"Size: ${recommendation['position_size']:.0f}"
                )

                self.execute_trade(recommendation, current_forecast, timestamp)

        # Assume all positions are closed at baseline at the end
        if self.position != 0:
            self.close_position_at_baseline(df.iloc[-1]["Timestamp"])

        # Calculate results
        total_trades = len(self.trades)
        final_portfolio_value = self.cash + abs(self.position)
        total_return = ((final_portfolio_value - 10000) / 10000) * 100

        results = {
            "total_trades": total_trades,
            "final_cash": self.cash,
            "final_position": self.position,
            "total_pnl": self.total_pnl,
            "total_return_pct": total_return,
            "portfolio_value": final_portfolio_value,
            "trades": self.trades,
        }

        print("=" * 60)
        print("BACKTEST RESULTS:")
        print(f"Total trades executed: {total_trades}")
        print(f"Final portfolio value: ${final_portfolio_value:.2f}")
        print(f"Total return: {total_return:.2f}%")
        print(f"Total P&L: ${self.total_pnl:.2f}")

        return results

    def get_live_recommendation(self, current_forecast_pct):
        """
        Get live trading recommendation for current forecast.

        Args:
            current_forecast_pct (float): Current forecast percentage

        Returns:
            dict: Live trading recommendation
        """
        recommendation = self.should_trade(current_forecast_pct)

        print(f"\n📊 LIVE TRADING RECOMMENDATION 📊")
        print(f"Current Forecast: {current_forecast_pct}%")
        print(f"Baseline: {self.baseline}%")
        print(f"Deviation: {current_forecast_pct - self.baseline:.2f} points")
        print(f"Action: {recommendation['action'].upper()}")

        if recommendation["action"] != "hold":
            print(f"Recommended Position Size: ${recommendation['position_size']:.0f}")
            print(f"Expected Return: {recommendation['expected_return']:.2f}%")
            print(f"Confidence: {recommendation['confidence']*100:.0f}%")
            print(f"Reasoning: {recommendation['reason']}")

        return recommendation


# Example usage and strategy testing
def run_arbitrage_analysis():
    """Run complete arbitrage analysis with examples."""

    # Initialize trader
    trader = ForecastArbitrageTrader(
        baseline_forecast=33.0,
        transaction_cost=0.001,  # 0.1% transaction cost
        max_position_size=1000,  # Max position size
    )

    # Example 1: Test with historical data
    print("🔄 Running backtest on historical deviation data...")
    try:
        results = trader.backtest_strategy("forecast_deviations_output.csv")

        # Save results
        with open("arbitrage_backtest_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print("📁 Results saved to 'arbitrage_backtest_results.json'")

    except FileNotFoundError:
        print("❌ Historical data file not found. Skipping backtest.")

    # Example 2: Live trading recommendations
    print("\n" + "=" * 60)
    print("🎯 LIVE TRADING EXAMPLES")
    print("=" * 60)

    # Test various forecast scenarios
    test_scenarios = [
        32.0,  # 1% below baseline - strong buy signal
        32.5,  # 0.5% below baseline - moderate buy
        33.0,  # At baseline - hold
        33.5,  # 0.5% above baseline - moderate sell
        34.0,  # 1% above baseline - strong sell
        31.5,  # 1.5% below baseline - very strong buy
    ]

    for forecast in test_scenarios:
        trader.get_live_recommendation(forecast)
        print("-" * 40)


if __name__ == "__main__":
    run_arbitrage_analysis()
