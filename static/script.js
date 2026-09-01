const form = document.getElementById("risk-form");
const resultEl = document.getElementById("result");
const labelEl = document.getElementById("result-label");
const pctEl = document.getElementById("result-pct");
const fillEl = document.getElementById("meter-fill");
const errorEl = document.getElementById("error");
const submitBtn = document.getElementById("submit-btn");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorEl.hidden = true;
  resultEl.hidden = true;

  const data = Object.fromEntries(new FormData(form).entries());

  submitBtn.disabled = true;
  submitBtn.textContent = "Estimating…";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const payload = await res.json();

    if (!res.ok) {
      errorEl.textContent = payload.error || "Something went wrong.";
      errorEl.hidden = false;
      return;
    }

    const pct = Math.round(payload.probability * 100);
    labelEl.textContent = payload.label;
    pctEl.textContent = `${pct}%`;
    fillEl.style.width = `${pct}%`;
    fillEl.style.background = payload.prediction === 1
      ? "var(--risk-high)"
      : "var(--risk-low)";

    resultEl.hidden = false;
  } catch (err) {
    errorEl.textContent = "Couldn't reach the server. Is app.py running?";
    errorEl.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Estimate risk";
  }
});
