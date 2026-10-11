"""Shared Plotly figures for the read-only research dashboard."""
import plotly.express as px


def line_chart(frame, x, y, title):
    """Render a compact interactive line chart."""
    return px.line(frame, x=x, y=y, title=title)


def forecast_chart(frame):
    """Plot actual and forecast volatility on comparable historical dates."""
    tidy = frame.melt(id_vars=["date", "model"], value_vars=["actual_volatility", "predicted_volatility"],
                      var_name="series", value_name="volatility")
    tidy["line"] = tidy["model"] + " · " + tidy["series"]
    return px.line(tidy, x="date", y="volatility", color="line", title="Test: dự báo và thực tế")
