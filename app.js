let latestData = null;
let historyData = null;

async function loadData() {
  latestData = await fetch("./data/latest.json").then(r => r.json());
  historyData = await fetch("./data/history.json").then(r => r.json());

  document.getElementById("updatedAt").textContent = latestData.updated_at || "-";
  initMarketSelector();
}

function initMarketSelector() {
  const select = document.getElementById("marketSelect");
  select.innerHTML = "";

  latestData.markets.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.symbol;
    opt.textContent = `${m.symbol} — ${m.name}`;
    select.appendChild(opt);
  });

  select.addEventListener("change", () => render(select.value));

  if (latestData.markets.length > 0) {
    render(latestData.markets[0].symbol);
  }
}

function render(symbol) {
  const latest = latestData.markets.find(m => m.symbol === symbol);
  const history = historyData[symbol] || [];

  if (!latest || history.length === 0) return;

  document.getElementById("priceValue").textContent = latest.price.toLocaleString();
  document.getElementById("stressRegime").textContent = latest.regime;
  document.getElementById("opportunityRegime").textContent = latest.opportunity_regime;
  document.getElementById("signalValue").textContent = latest.signal;

  renderGauge("stressGauge", latest.fear_index, "stress");
  renderGauge("opportunityGauge", latest.opportunity_index, "opportunity");
  renderChart(history, symbol);
}

function renderGauge(id, value, mode) {
  const isStress = mode === "stress";

  const steps = isStress
    ? [
        { range: [0, 25], color: "#d9f2e3" },
        { range: [25, 50], color: "#f6e58d" },
        { range: [50, 75], color: "#f8c291" },
        { range: [75, 100], color: "#f46b6b" }
      ]
    : [
        { range: [0, 25], color: "#dbeafe" },
        { range: [25, 50], color: "#bfdbfe" },
        { range: [50, 75], color: "#93c5fd" },
        { range: [75, 100], color: "#3b82f6" }
      ];

  Plotly.newPlot(
    id,
    [{
      type: "indicator",
      mode: "gauge+number",
      value: value,
      number: {
        font: {
          size: 40,
          color: "#0f172a"
        }
      },
      gauge: {
        shape: "angular",
        axis: {
          range: [0, 100],
          tickmode: "array",
          tickvals: [0, 25, 50, 75, 100],
          ticktext: ["0", "25", "50", "75", "100"],
          tickfont: {
            size: 13,
            color: "#64748b"
          }
        },
        // Hide the continuous progress bar to act purely as a background
        bar: {
          color: "rgba(0,0,0,0)",
          thickness: 0
        },
        bgcolor: "#ffffff",
        borderwidth: 0,
        steps: steps,
        // Repurpose threshold as the pointer line
        threshold: {
          line: {
            color: "#0f172a",
            width: 5
          },
          thickness: 1, // Extend the pointer across the entire track width
          value: value
        }
      }
    }],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      margin: { t: 10, r: 25, l: 25, b: 10 }
    },
    {
      responsive: true,
      displayModeBar: false
    }
  );
}

function renderChart(history, symbol) {
  // 1. Strict Date Formatting: Ensure 'YYYY-MM-DD' with zero-padding
  // This prevents Plotly from miscalculating the axis bounds due to string comparison errors.
  const dates = history.map(d => {
    const parts = d.date.split("/");
    if (parts.length === 3) {
      const y = parts[0];
      const m = String(parts[1]).padStart(2, "0");
      const day = String(parts[2]).padStart(2, "0");
      return `${y}-${m}-${day}`;
    }
    return d.date; 
  });

  const price = history.map(d => d.price);
  const fear = history.map(d => d.fear_index);
  const opp = history.map(d => d.opportunity_index);

  // 2. Identify absolute boundaries of the dataset
  const firstDate = dates[0];
  const latestDate = dates[dates.length - 1];

  // 3. Define the default visible range for the main chart (e.g., the most recent 1 year)
  // This highlights the recent trend directly upon loading.
  const endObj = new Date(latestDate);
  const startObj = new Date(endObj);
  startObj.setFullYear(startObj.getFullYear() - 1); 
  const recentStartDate = startObj.toISOString().split("T")[0];

  const traces = [
    {
      x: dates,
      y: price,
      mode: "lines",
      name: `${symbol} Price`,
      yaxis: "y1",
      line: { width: 3, color: "#2563eb" },
      hovertemplate: `${symbol}=%{y:.2f}<extra></extra>`
    },
    {
      x: dates,
      y: fear,
      mode: "lines",
      name: "Stress Index",
      yaxis: "y2",
      line: { width: 2.3, color: "#ef4444" },
      hovertemplate: `Stress=%{y:.2f}<extra></extra>`
    },
    {
      x: dates,
      y: opp,
      mode: "lines",
      name: "Opportunity Index",
      yaxis: "y2",
      line: { width: 2.3, color: "#10b981", dash: "dot" },
      hovertemplate: `Opportunity=%{y:.2f}<extra></extra>`
    }
  ];

  const layout = {
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    hovermode: "x unified",
    margin: { t: 70, r: 80, l: 75, b: 70 },
    legend: {
      orientation: "h",
      y: 1.10,
      yanchor: "bottom",
      x: 0.5,
      xanchor: "center",
      font: { color: "#334155", size: 13 }
    },
    xaxis: {
      type: "date",
      title: {
        text: "Date",
        font: { color: "#334155" }
      },
      tickformat: "%Y/%m/%d",
      // 4. Set main chart range to recent trends (disabling buggy autorange)
      range: [recentStartDate, latestDate],
      autorange: false,
      rangeslider: {
        visible: true,
        // 5. Set rangeslider to explicitly encompass the entire historical dataset
        range: [firstDate, latestDate],
        autorange: false,
        bgcolor: "#eef4fb",
        bordercolor: "#dbe6f2",
        borderwidth: 1
      },
      gridcolor: "#e5edf6",
      zerolinecolor: "#e5edf6",
      color: "#334155"
    },
    yaxis: {
      title: {
        text: "Price",
        font: { color: "#334155" }
      },
      side: "left",
      gridcolor: "#e5edf6",
      zerolinecolor: "#e5edf6",
      color: "#334155"
    },
    yaxis2: {
      title: {
        text: "Index (0-100)",
        font: { color: "#334155" }
      },
      overlaying: "y",
      side: "right",
      range: [0, 100],
      showgrid: false,
      color: "#334155"
    }
  };

  Plotly.newPlot("chart", traces, layout, {
    responsive: true,
    displaylogo: false
  });
}

loadData();