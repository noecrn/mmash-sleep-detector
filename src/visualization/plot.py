import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

def plot_user_hr(user_id: str):
    """
    Plot heart rate (hr_mean) over time for a given user.
    """
    path = Path(f"../data/processed/{user_id}.csv")
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    df = pd.read_csv(path, parse_dates=["timestamp"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["bpm"],
        mode="lines",
        name="Heart Rate (bpm)",
        line=dict(color="royalblue")
    ))

    fig.update_layout(
        title=f"Heart Rate over Time – {user_id}",
        xaxis_title="Timestamp",
        yaxis_title="Heart Rate (bpm)",
        hovermode="x unified"
    )

    fig.show()