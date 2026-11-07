"use strict";

document.addEventListener("DOMContentLoaded", () => {
    initializeNavigation();
    initializeOptimization();
    resizeExistingCharts();

    if (window.lucide) {
        window.lucide.createIcons();
    }
});


function initializeNavigation() {
    const toggle = document.getElementById("menuToggle");
    const sidebar = document.getElementById("sidebar");

    if (!toggle || !sidebar) {
        return;
    }

    toggle.addEventListener("click", () => {
        sidebar.classList.toggle("open");
    });

    document.addEventListener("click", (event) => {
        if (
            window.innerWidth <= 800 &&
            sidebar.classList.contains("open") &&
            !sidebar.contains(event.target) &&
            !toggle.contains(event.target)
        ) {
            sidebar.classList.remove("open");
        }
    });
}


function resizeExistingCharts() {
    if (!window.Plotly) {
        return;
    }

    window.addEventListener("resize", () => {
        document.querySelectorAll(".js-plotly-plot").forEach((chart) => {
            window.Plotly.Plots.resize(chart);
        });
    });
}


function initializeOptimization() {
    const form = document.getElementById("optimizationForm");

    if (!form) {
        return;
    }

    const button = document.getElementById("runOptimization");
    const errorBox = document.getElementById("optimizationError");
    const resultsSection = document.getElementById("optimizationResults");
    const downloadButton = document.getElementById("downloadAllocation");

    let latestResult = null;

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        errorBox.hidden = true;
        errorBox.textContent = "";

        const payload = {
            objective: document.getElementById("objective").value,
            lookback: Number(document.getElementById("lookback").value),
            risk_free_rate:
                Number(document.getElementById("riskFreeRate").value) / 100,
            min_weight:
                Number(document.getElementById("minWeight").value) / 100,
            max_weight:
                Number(document.getElementById("maxWeight").value) / 100,
            frontier_points: Number(
                document.getElementById("frontierPoints").value
            )
        };

        const validationError = validatePayload(payload);

        if (validationError) {
            showError(errorBox, validationError);
            return;
        }

        setLoading(button, true);

        try {
            const response = await fetch("/api/optimization", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (!response.ok || result.status !== "success") {
                throw new Error(
                    result.message || "The optimization request failed."
                );
            }

            latestResult = result;

            renderOptimizationResults(result, resultsSection);

            resultsSection.hidden = false;
            resultsSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

            if (window.lucide) {
                window.lucide.createIcons();
            }

        } catch (error) {
            showError(
                errorBox,
                error.message || "Unable to calculate the portfolio."
            );
        } finally {
            setLoading(button, false);
        }
    });

    if (downloadButton) {
        downloadButton.addEventListener("click", () => {
            if (!latestResult || !latestResult.allocation) {
                return;
            }

            downloadAllocationCsv(latestResult);
        });
    }
}


function validatePayload(payload) {
    if (!Number.isInteger(payload.lookback) || payload.lookback < 63) {
        return "Lookback must be at least 63 trading days.";
    }

    if (!Number.isFinite(payload.risk_free_rate)) {
        return "Enter a valid risk-free rate.";
    }

    if (
        payload.risk_free_rate < -0.05 ||
        payload.risk_free_rate > 0.25
    ) {
        return "Risk-free rate must be between -5% and 25%.";
    }

    if (
        !Number.isFinite(payload.min_weight) ||
        payload.min_weight < 0 ||
        payload.min_weight > 1
    ) {
        return "Minimum weight must be between 0% and 100%.";
    }

    if (
        !Number.isFinite(payload.max_weight) ||
        payload.max_weight <= 0 ||
        payload.max_weight > 1
    ) {
        return "Maximum weight must be greater than 0% and at most 100%.";
    }

    if (payload.min_weight > payload.max_weight) {
        return "Minimum weight cannot exceed maximum weight.";
    }

    if (
        !Number.isInteger(payload.frontier_points) ||
        payload.frontier_points < 10 ||
        payload.frontier_points > 60
    ) {
        return "Frontier points must be between 10 and 60.";
    }

    return null;
}


function setLoading(button, loading) {
    if (!button) {
        return;
    }

    button.disabled = loading;

    if (loading) {
        button.innerHTML = `
            <span class="loading-spinner"></span>
            Calculating portfolio...
        `;
    } else {
        button.innerHTML = `
            <i data-lucide="play"></i>
            Calculate portfolio
        `;

        if (window.lucide) {
            window.lucide.createIcons();
        }
    }
}


function showError(element, message) {
    if (!element) {
        return;
    }

    element.textContent = message;
    element.hidden = false;
}


function formatPercent(value, digits = 2) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return `${(number * 100).toFixed(digits)}%`;
}


function formatNumber(value, digits = 3) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return number.toFixed(digits);
}


function renderOptimizationResults(result, resultsSection) {
    const metrics = result.selected_metrics;
    const estimation = result.estimation;

    document.getElementById("metricReturn").textContent =
        formatPercent(metrics.expected_annual_return);

    document.getElementById("metricVolatility").textContent =
        formatPercent(metrics.annual_volatility);

    document.getElementById("metricSharpe").textContent =
        formatNumber(metrics.sharpe_ratio);

    const activeAssets = result.allocation.filter(
        (item) => item.weight > 0.000001
    ).length;

    document.getElementById("metricAssets").textContent =
        String(activeAssets);

    document.getElementById("resultPeriod").textContent =
        `${estimation.start_date} to ${estimation.end_date} · ` +
        `${estimation.observations.toLocaleString()} observations · ` +
        `${estimation.asset_count} assets`;

    renderFrontierChart(result);
    renderAllocationChart(result);
    renderRiskContributionChart(result);
    renderCorrelationChart(result);
    renderAllocationTable(result);
    renderDiagnostics(result);

    resultsSection.hidden = false;

    window.setTimeout(() => {
        document.querySelectorAll(
            "#optimizationResults .js-plotly-plot"
        ).forEach((chart) => {
            window.Plotly.Plots.resize(chart);
        });
    }, 100);
}


function renderFrontierChart(result) {
    const points = result.efficient_frontier || [];
    const maxSharpe = result.maximum_sharpe_metrics;
    const minVolatility = result.minimum_volatility_metrics;
    const selected = result.selected_metrics;

    const traces = [
        {
            x: points.map((point) => point.annual_volatility * 100),
            y: points.map((point) => point.expected_annual_return * 100),
            mode: "lines+markers",
            type: "scatter",
            name: "Efficient frontier",
            line: {
                color: "#4263eb",
                width: 3,
                shape: "spline"
            },
            marker: {
                size: 7,
                color: points.map((point) => point.sharpe_ratio),
                colorscale: "Viridis",
                showscale: true,
                colorbar: {
                    title: "Sharpe",
                    thickness: 10
                }
            },
            customdata: points.map((point) => [
                point.sharpe_ratio
            ]),
            hovertemplate:
                "Volatility: %{x:.2f}%<br>" +
                "Expected return: %{y:.2f}%<br>" +
                "Sharpe: %{customdata[0]:.3f}<extra></extra>"
        },
        {
            x: [maxSharpe.annual_volatility * 100],
            y: [maxSharpe.expected_annual_return * 100],
            mode: "markers",
            type: "scatter",
            name: "Maximum Sharpe",
            marker: {
                size: 15,
                color: "#f08c32",
                symbol: "star",
                line: { color: "#ffffff", width: 2 }
            },
            hovertemplate:
                "Maximum Sharpe<br>" +
                "Volatility: %{x:.2f}%<br>" +
                "Return: %{y:.2f}%<extra></extra>"
        },
        {
            x: [minVolatility.annual_volatility * 100],
            y: [minVolatility.expected_annual_return * 100],
            mode: "markers",
            type: "scatter",
            name: "Minimum volatility",
            marker: {
                size: 13,
                color: "#0ca678",
                symbol: "diamond",
                line: { color: "#ffffff", width: 2 }
            },
            hovertemplate:
                "Minimum volatility<br>" +
                "Volatility: %{x:.2f}%<br>" +
                "Return: %{y:.2f}%<extra></extra>"
        },
        {
            x: [selected.annual_volatility * 100],
            y: [selected.expected_annual_return * 100],
            mode: "markers",
            type: "scatter",
            name: "Selected objective",
            marker: {
                size: 17,
                color: "#845ef7",
                symbol: "circle-open",
                line: { width: 3 }
            },
            hovertemplate:
                "Selected portfolio<br>" +
                "Volatility: %{x:.2f}%<br>" +
                "Return: %{y:.2f}%<extra></extra>"
        }
    ];

    const layout = baseChartLayout("Efficient frontier");

    layout.xaxis = {
        title: "Annualized volatility (%)",
        ticksuffix: "%",
        gridcolor: "#edf1f7",
        zeroline: false
    };

    layout.yaxis = {
        title: "Expected annual return (%)",
        ticksuffix: "%",
        gridcolor: "#edf1f7",
        zeroline: false
    };

    Plotly.newPlot("frontierChart", traces, layout, chartConfig());
}


function renderAllocationChart(result) {
    const allocation = result.allocation.filter(
        (item) => item.weight > 0.000001
    );

    const trace = {
        type: "pie",
        labels: allocation.map((item) => item.asset),
        values: allocation.map((item) => item.weight_pct),
        hole: 0.56,
        textinfo: "label+percent",
        textposition: "outside",
        sort: false,
        marker: {
            line: {
                color: "#ffffff",
                width: 2
            }
        },
        hovertemplate:
            "<b>%{label}</b><br>Allocation: %{value:.2f}%<extra></extra>"
    };

    const layout = baseChartLayout("Portfolio weights");
    layout.showlegend = true;
    layout.legend = {
        orientation: "h",
        y: -0.12,
        x: 0.5,
        xanchor: "center"
    };

    Plotly.newPlot(
        "allocationChart",
        [trace],
        layout,
        chartConfig()
    );
}


function renderRiskContributionChart(result) {
    const contributions = [...result.risk_contributions].sort(
        (a, b) => a.risk_contribution_pct - b.risk_contribution_pct
    );

    const trace = {
        type: "bar",
        orientation: "h",
        x: contributions.map(
            (item) => item.risk_contribution_pct * 100
        ),
        y: contributions.map((item) => item.asset),
        marker: {
            color: contributions.map(
                (item) => item.risk_contribution_pct * 100
            ),
            colorscale: "Teal",
            showscale: false
        },
        text: contributions.map(
            (item) =>
                `${(item.risk_contribution_pct * 100).toFixed(2)}%`
        ),
        textposition: "outside",
        hovertemplate:
            "<b>%{y}</b><br>Risk contribution: %{x:.2f}%<extra></extra>"
    };

    const layout = baseChartLayout("Risk contribution by asset");

    layout.xaxis = {
        title: "Contribution to portfolio volatility (%)",
        ticksuffix: "%",
        gridcolor: "#edf1f7"
    };

    layout.yaxis = {
        title: "",
        automargin: true
    };

    Plotly.newPlot(
        "riskContributionChart",
        [trace],
        layout,
        chartConfig()
    );
}


function renderCorrelationChart(result) {
    const correlation = result.correlation;

    const trace = {
        type: "heatmap",
        x: correlation.assets,
        y: correlation.assets,
        z: correlation.values,
        zmin: -1,
        zmax: 1,
        colorscale: [
            [0, "#e85959"],
            [0.5, "#ffffff"],
            [1, "#4263eb"]
        ],
        colorbar: {
            title: "Correlation",
            thickness: 12
        },
        hovertemplate:
            "%{y} / %{x}<br>Correlation: %{z:.3f}<extra></extra>"
    };

    const layout = baseChartLayout("Historical asset correlation");
    layout.height = 460;
    layout.margin = { l: 75, r: 35, t: 25, b: 75 };
    layout.xaxis = {
        tickangle: -45,
        side: "bottom",
        automargin: true
    };
    layout.yaxis = {
        automargin: true,
        autorange: "reversed"
    };

    Plotly.newPlot(
        "correlationChart",
        [trace],
        layout,
        chartConfig()
    );
}


function renderAllocationTable(result) {
    const tbody = document.querySelector("#allocationTable tbody");
    const riskLookup = new Map(
        result.risk_contributions.map((item) => [
            item.asset,
            item.risk_contribution_pct
        ])
    );

    tbody.replaceChildren();

    const sortedAllocation = [...result.allocation].sort(
        (a, b) => b.weight - a.weight
    );

    sortedAllocation.forEach((item) => {
        const row = document.createElement("tr");

        const values = [
            item.asset,
            formatPercent(item.weight),
            formatPercent(item.expected_return),
            formatPercent(item.annual_volatility),
            formatPercent(riskLookup.get(item.asset))
        ];

        values.forEach((value, index) => {
            const cell = document.createElement("td");
            cell.textContent = value;

            if (index === 0) {
                cell.style.fontWeight = "700";
                cell.style.color = "#344054";
            }

            row.appendChild(cell);
        });

        tbody.appendChild(row);
    });
}


function renderDiagnostics(result) {
    const container = document.getElementById(
        "optimizationDiagnostics"
    );

    const estimation = result.estimation;

    const items = [
        ["Objective", result.objective.replaceAll("_", " ")],
        ["Estimation method", estimation.method],
        ["Lookback", `${estimation.lookback_days} trading days`],
        ["Observations", estimation.observations.toLocaleString()],
        ["Assets", estimation.asset_count],
        ["Risk-free rate", formatPercent(estimation.risk_free_rate)],
        ["Minimum weight", formatPercent(estimation.min_weight)],
        ["Maximum weight", formatPercent(estimation.max_weight)],
        ["Estimation period", `${estimation.start_date} to ${estimation.end_date}`],
        ["Portfolio construction", estimation.rebalancing],
        ["Frontier portfolios", result.efficient_frontier.length],
        ["Weight convention", "Long-only, fully invested"]
    ];

    container.replaceChildren();

    items.forEach(([label, value]) => {
        const item = document.createElement("div");
        item.className = "diagnostic-item";

        const title = document.createElement("span");
        title.textContent = label;

        const content = document.createElement("strong");
        content.textContent = String(value);

        item.append(title, content);
        container.appendChild(item);
    });
}


function baseChartLayout(title) {
    return {
        title: {
            text: title,
            font: {
                family: "DM Sans, sans-serif",
                size: 13,
                color: "#344054"
            },
            x: 0.02,
            xanchor: "left"
        },
        autosize: true,
        height: 350,
        margin: {
            l: 55,
            r: 30,
            t: 45,
            b: 55
        },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {
            family: "DM Sans, sans-serif",
            color: "#667085",
            size: 10
        },
        hoverlabel: {
            bgcolor: "#ffffff",
            font: {
                color: "#182230"
            }
        },
        legend: {
            orientation: "h",
            y: -0.15
        }
    };
}


function chartConfig() {
    return {
        responsive: true,
        displaylogo: false,
        scrollZoom: true,
        modeBarButtonsToRemove: [
            "lasso2d",
            "select2d"
        ]
    };
}


function downloadAllocationCsv(result) {
    const headers = [
        "asset",
        "weight",
        "weight_pct",
        "expected_annual_return",
        "annual_volatility",
        "risk_contribution_pct"
    ];

    const riskLookup = new Map(
        result.risk_contributions.map((item) => [
            item.asset,
            item.risk_contribution_pct
        ])
    );

    const rows = result.allocation.map((item) => [
        item.asset,
        item.weight,
        item.weight_pct,
        item.expected_return,
        item.annual_volatility,
        riskLookup.get(item.asset)
    ]);

    const escapeCsv = (value) => {
        const text = String(value ?? "");
        return `"${text.replaceAll('"', '""')}"`;
    };

    const csv = [
        headers.map(escapeCsv).join(","),
        ...rows.map((row) => row.map(escapeCsv).join(","))
    ].join("\r\n");

    const blob = new Blob(
        ["\uFEFF" + csv],
        { type: "text/csv;charset=utf-8;" }
    );

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "optimized_portfolio_allocation.csv";

    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(url);
}

const spinnerStyles = document.createElement("style");
spinnerStyles.textContent = `
    .loading-spinner {
        width: 15px;
        height: 15px;
        border: 2px solid rgba(255, 255, 255, 0.4);
        border-top-color: #ffffff;
        border-radius: 50%;
        animation: spinner-rotate 0.7s linear infinite;
    }

    @keyframes spinner-rotate {
        to {
            transform: rotate(360deg);
        }
    }
`;
document.head.appendChild(spinnerStyles);