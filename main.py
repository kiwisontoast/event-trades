import pandas as pd
from collections import Counter
from datetime import datetime
import numpy as np


def analyze_forecast_deviations(csv_file_path):
    """
    Analyze forecast data to identify baseline and deviations.

    Args:
        csv_file_path (str): Path to the CSV file

    Returns:
        dict: Analysis results including baseline, deviations, and statistics
    """

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Convert percentage strings to float values
    df["Forecast_Value"] = df["Forecast"].str.rstrip("%").astype(float)

    # Convert timestamp to datetime

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Find the baseline (most frequent forecast value)
    forecast_counts = df["Forecast_Value"].value_counts()
    baseline_value = forecast_counts.index[0]  # Most frequent value
    baseline_count = forecast_counts.iloc[0]

    print(f"Baseline Analysis:")
    print(f"- Baseline value: {baseline_value}%")
    print(f"- Occurrences: {baseline_count} out of {len(df)} total records")
    print(f"- Percentage of time at baseline: {(baseline_count/len(df)*100):.2f}%")
    print()

    # Identify deviations from baseline
    deviations = df[df["Forecast_Value"] != baseline_value].copy()

    # Calculate deviation amount and direction
    deviations["Deviation_Amount"] = deviations["Forecast_Value"] - baseline_value
    deviations["Direction"] = deviations["Deviation_Amount"].apply(
        lambda x: "UP" if x > 0 else "DOWN"
    )

    # Sort by timestamp for chronological analysis
    deviations = deviations.sort_values("Timestamp")

    print(f"Deviation Analysis:")
    print(f"- Total deviations: {len(deviations)}")
    print(f"- Upward deviations: {len(deviations[deviations['Direction'] == 'UP'])}")
    print(
        f"- Downward deviations: {len(deviations[deviations['Direction'] == 'DOWN'])}"
    )
    print()

    # Display deviation summary
    if len(deviations) > 0:
        print("Deviation Summary:")
        deviation_summary = (
            deviations.groupby(["Forecast_Value", "Direction"])
            .agg({"Deviation_Amount": ["count", "first"], "Timestamp": ["min", "max"]})
            .round(4)
        )

        print(deviation_summary)
        print()

        # Show detailed deviation records
        print("Detailed Deviation Records:")
        print("=" * 80)
        for idx, row in deviations.iterrows():
            print(f"Time: {row['Timestamp']}")
            print(
                f"Forecast: {row['Forecast_Value']}% ({row['Direction']} by {abs(row['Deviation_Amount']):.2f} points)"
            )
            print(f"Market: {row['Market']}")
            print("-" * 40)

    # Create summary statistics
    if len(deviations) > 0:
        stats = {
            "baseline_value": baseline_value,
            "baseline_percentage": (baseline_count / len(df) * 100),
            "total_deviations": len(deviations),
            "max_upward_deviation": (
                deviations[deviations["Direction"] == "UP"]["Deviation_Amount"].max()
                if len(deviations[deviations["Direction"] == "UP"]) > 0
                else 0
            ),
            "max_downward_deviation": (
                abs(
                    deviations[deviations["Direction"] == "DOWN"][
                        "Deviation_Amount"
                    ].min()
                )
                if len(deviations[deviations["Direction"] == "DOWN"]) > 0
                else 0
            ),
            "avg_deviation_magnitude": abs(deviations["Deviation_Amount"]).mean(),
            "deviations_by_hour": deviations.groupby(deviations["Timestamp"].dt.hour)
            .size()
            .to_dict(),
        }

        print("\nSummary Statistics:")
        print(f"- Baseline: {stats['baseline_value']:.2f}%")
        print(f"- Time at baseline: {stats['baseline_percentage']:.2f}%")
        print(f"- Total deviations: {stats['total_deviations']}")
        print(
            f"- Largest upward deviation: +{stats['max_upward_deviation']:.2f} points"
        )
        print(
            f"- Largest downward deviation: -{stats['max_downward_deviation']:.2f} points"
        )
        print(
            f"- Average deviation magnitude: {stats['avg_deviation_magnitude']:.2f} points"
        )

        return deviations, stats
    else:
        print("No deviations found from baseline.")
        return pd.DataFrame(), {"baseline_value": baseline_value}


def export_deviations(deviations_df, output_file="forecast_deviations.csv"):
    """
    Export deviation records to a CSV file.

    Args:
        deviations_df (DataFrame): DataFrame containing deviation records
        output_file (str): Output file name
    """
    if len(deviations_df) > 0:
        # Select relevant columns for export
        export_df = deviations_df[
            [
                "Timestamp",
                "Market",
                "Forecast",
                "Forecast_Value",
                "Deviation_Amount",
                "Direction",
            ]
        ].copy()

        # Rename columns for clarity
        export_df.columns = [
            "Timestamp",
            "Market",
            "Original_Forecast",
            "Forecast_Value",
            "Deviation_Points",
            "Direction",
        ]

        export_df.to_csv(output_file, index=False)
        print(f"\nDeviation records exported to: {output_file}")
    else:
        print("No deviations to export.")


# Example usage:
if __name__ == "__main__":
    # Replace with your actual file path
    csv_file_path = "oklahomavsindiana4.csv"

    try:
        # Analyze the data
        deviations_df, stats = analyze_forecast_deviations(csv_file_path)

        # Export deviations to a new file
        export_deviations(deviations_df, "forecast_deviations_output.csv")

    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file_path}'")
        print("Please make sure the file path is correct.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")
