"""
Medical Visualizer — creates Plotly charts for the clinical dashboard.
Generates feature importance bars, confidence gauges, and SHAP summaries.
"""
import logging

logger = logging.getLogger(__name__)


class MedicalVisualizer:

    @staticmethod
    def feature_importance_chart(shap_values: dict, title: str = "Feature Contributions"):
        """Horizontal bar chart coloured by direction of contribution."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("plotly not installed.")
            return None

        features = list(shap_values.keys())
        values   = list(shap_values.values())
        colors   = ["#FC8181" if v > 0 else "#68D391" for v in values]

        fig = go.Figure(go.Bar(
            x=values, y=features, orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            title=title,
            xaxis_title="SHAP Contribution",
            yaxis_title="Feature",
            height=max(300, len(features) * 40),
            plot_bgcolor="#111827",
            paper_bgcolor="#111827",
            font={"color": "#E2E8F0"},
            margin={"l": 160, "r": 60, "t": 50, "b": 40},
        )
        return fig

    @staticmethod
    def confidence_gauge(confidence: float, title: str = "Prediction Confidence"):
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        color = "#FC8181" if confidence > 0.65 else ("#F6AD55" if confidence > 0.4 else "#68D391")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(confidence * 100, 1),
            number={"suffix": "%"},
            title={"text": title, "font": {"color": "#E2E8F0"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#718096"},
                "bar": {"color": color},
                "bgcolor": "#1A2236",
                "bordercolor": "#2D3748",
                "steps": [
                    {"range": [0, 40],  "color": "#1A3A1A"},
                    {"range": [40, 65], "color": "#3A2E10"},
                    {"range": [65, 100],"color": "#3A1010"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": round(confidence * 100, 1),
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="#111827",
            font={"color": "#E2E8F0"},
            height=250,
            margin={"t": 60, "b": 20, "l": 20, "r": 20},
        )
        return fig

    @staticmethod
    def uncertainty_plot(predictions_bootstrap: list, title: str = "Bootstrap Uncertainty"):
        try:
            import plotly.graph_objects as go
            import numpy as np
        except ImportError:
            return None

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=predictions_bootstrap,
            nbinsx=30,
            marker_color="#63B3ED",
            opacity=0.75,
            name="Bootstrap predictions",
        ))
        mean_val = float(np.mean(predictions_bootstrap))
        fig.add_vline(x=mean_val, line_color="white", line_dash="dash",
                      annotation_text=f"Mean: {mean_val:.2f}")
        fig.update_layout(
            title=title,
            xaxis_title="Predicted Probability",
            yaxis_title="Frequency",
            plot_bgcolor="#111827",
            paper_bgcolor="#111827",
            font={"color": "#E2E8F0"},
            height=260,
        )
        return fig

    @staticmethod
    def fairness_chart(group_metrics: dict):
        """Grouped bar chart of sensitivity/specificity by demographic group."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        groups = list(group_metrics.keys())
        sensitivity = [group_metrics[g].get("sensitivity", 0) for g in groups]
        specificity  = [group_metrics[g].get("specificity", 0) for g in groups]

        fig = go.Figure(data=[
            go.Bar(name="Sensitivity", x=groups, y=sensitivity,
                   marker_color="#63B3ED"),
            go.Bar(name="Specificity", x=groups, y=specificity,
                   marker_color="#4FD1C5"),
        ])
        fig.update_layout(
            barmode="group",
            title="Fairness Monitor — Group Performance",
            yaxis={"range": [0, 1], "tickformat": ".0%"},
            plot_bgcolor="#111827",
            paper_bgcolor="#111827",
            font={"color": "#E2E8F0"},
            height=300,
        )
        return fig
