let currentChart = null;
let currentTableItems = [];

export function injectFavicon() {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" rx="22" fill="#0f2b5c"/><path d="M75 35 A 30 30 0 1 0 75 65" stroke="#ffffff" stroke-width="8" stroke-linecap="round"/><path d="M35 50 L50 65 L80 30" stroke="#2dd4bf" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
    const link = document.createElement('link');
    link.rel = 'icon';
    link.type = 'image/svg+xml';
    link.href = 'data:image/svg+xml;base64,' + btoa(svg);
    document.head.appendChild(link);
}

export function showToast(message, type = 'info') {
    let container = document.getElementById('toastContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container-custom';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `toast-custom toast-${type}`;

    let iconClass = 'fa-circle-info text-info';
    if (type === 'success') iconClass = 'fa-circle-check text-success';
    if (type === 'danger') iconClass = 'fa-triangle-exclamation text-danger';
    if (type === 'warning') iconClass = 'fa-triangle-exclamation text-warning';

    toast.innerHTML = `
        <div class="d-flex align-items-center gap-2">
            <i class="fas ${iconClass} fa-lg"></i>
            <span style="font-size: 0.9rem;">${message}</span>
        </div>
        <button class="btn-close btn-close-sm ms-2" onclick="this.parentElement.remove()"></button>
    `;

    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(50px)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

export function customConfirm(message, title = "Conferma Operazione") {
    return new Promise((resolve) => {
        const modal = document.getElementById('confirmModal');
        const titleEl = document.getElementById('confirmModalTitle');
        const msgEl = document.getElementById('confirmModalMessage');
        const btnYes = document.getElementById('btnConfirmYes');
        const btnNo = document.getElementById('btnConfirmNo');

        titleEl.textContent = title;
        msgEl.textContent = message;
        modal.classList.remove('d-none');

        const cleanup = () => {
            modal.classList.add('d-none');
            btnYes.removeEventListener('click', onYes);
            btnNo.removeEventListener('click', onNo);
        };

        const onYes = () => { cleanup(); resolve(true); };
        const onNo = () => { cleanup(); resolve(false); };

        btnYes.addEventListener('click', onYes);
        btnNo.addEventListener('click', onNo);
    });
}

export function customPrompt(message, defaultValue = "", title = "Inserisci Valore") {
    return new Promise((resolve) => {
        const modal = document.getElementById('promptModal');
        const titleEl = document.getElementById('promptModalTitle');
        const msgEl = document.getElementById('promptModalMessage');
        const inputEl = document.getElementById('promptModalInput');
        const btnOk = document.getElementById('btnPromptOk');
        const btnCancel = document.getElementById('btnPromptCancel');

        titleEl.textContent = title;
        msgEl.textContent = message;
        inputEl.value = defaultValue;
        modal.classList.remove('d-none');
        inputEl.focus();

        const cleanup = () => {
            modal.classList.add('d-none');
            btnOk.removeEventListener('click', onOk);
            btnCancel.removeEventListener('click', onCancel);
        };

        const onOk = () => { const val = inputEl.value; cleanup(); resolve(val); };
        const onCancel = () => { cleanup(); resolve(null); };

        btnOk.addEventListener('click', onOk);
        btnCancel.addEventListener('click', onCancel);
    });
}

export function renderKPIDashboard(stats) {
    const dashboard = document.getElementById('kpiDashboard');
    if (!dashboard || !stats) return;

    const debPerc = stats._raw_debit_amount_perc || 0;
    const credPerc = stats._raw_credit_amount_perc || 0;
    const unreconciledDeb = stats["Unused Receipts (DEBIT)"] || 0;
    const unreconciledCred = stats["Unreconciled Deposits (CREDIT)"] || 0;
    const finalDelta = stats["Final delta (DEBIT - CREDIT)"] || "0.00 €";

    const debClass = debPerc >= 95 ? "text-success" : (debPerc >= 70 ? "text-warning" : "text-danger");
    const credClass = credPerc >= 95 ? "text-success" : (credPerc >= 70 ? "text-warning" : "text-danger");
    const unrecClass = (unreconciledDeb + unreconciledCred) === 0 ? "text-success" : "text-warning";

    dashboard.innerHTML = `
        <div class="kpi-card">
            <div class="kpi-title"><i class="fas fa-arrow-down-left text-success me-1"></i>Copertura Incassi (Dare)</div>
            <div class="kpi-value ${debClass}">${debPerc.toFixed(1)}%</div>
            <div class="kpi-sub">${stats["Used Receipts (DEBIT)"]} su ${stats["Total Receipts (DEBIT)"]} usati</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title"><i class="fas fa-arrow-up-right text-primary me-1"></i>Copertura Versamenti (Avere)</div>
            <div class="kpi-value ${credClass}">${credPerc.toFixed(1)}%</div>
            <div class="kpi-sub">${stats["Reconciled Deposits (CREDIT)"]} su ${stats["Total Deposits (CREDIT)"]} abbinati</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title"><i class="fas fa-triangle-exclamation me-1"></i>Non Riconciliati</div>
            <div class="kpi-value ${unrecClass}">${unreconciledDeb + unreconciledCred}</div>
            <div class="kpi-sub">${unreconciledDeb} incassi, ${unreconciledCred} versamenti</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title"><i class="fas fa-scale-balanced me-1"></i>Delta Finale</div>
            <div class="kpi-value">${finalDelta}</div>
            <div class="kpi-sub">Differenza residua (Dare − Avere)</div>
        </div>
    `;
}

export function renderMatchBadges(matchCounts) {
    const container = document.getElementById('matchBadgesSummary');
    if (!container || !matchCounts) return;

    container.innerHTML = `
        <span class="badge bg-success-subtle text-success border border-success-subtle rounded-pill px-3 py-2 fw-semibold">
            <i class="fas fa-check-double me-1"></i>${matchCounts.exact} Match Esatti
        </span>
        ${matchCounts.tolerance > 0 ? `
        <span class="badge bg-info-subtle text-info border border-info-subtle rounded-pill px-3 py-2 fw-semibold">
            <i class="fas fa-arrows-left-right me-1"></i>${matchCounts.tolerance} Match con Tolleranza
        </span>` : ''}
        ${matchCounts.anomaly > 0 ? `
        <span class="badge bg-danger-subtle text-danger border border-danger-subtle rounded-pill px-3 py-2 fw-semibold">
            <i class="fas fa-triangle-exclamation me-1"></i>${matchCounts.anomaly} Anomalie
        </span>` : ''}
        ${matchCounts.unused_debit > 0 ? `
        <span class="badge bg-warning-subtle text-warning border border-warning-subtle rounded-pill px-3 py-2 fw-semibold">
            <i class="fas fa-arrow-down-left me-1"></i>${matchCounts.unused_debit} Incassi non usati
        </span>` : ''}
        ${matchCounts.unreconciled_credit > 0 ? `
        <span class="badge bg-secondary-subtle text-secondary border border-secondary-subtle rounded-pill px-3 py-2 fw-semibold">
            <i class="fas fa-arrow-up-right me-1"></i>${matchCounts.unreconciled_credit} Versamenti scoperti
        </span>` : ''}
    `;
}

export function renderMonthlyChart(monthlyBalance) {
    const canvas = document.getElementById('monthlyChart');
    if (!canvas || !monthlyBalance || monthlyBalance.length === 0) return;

    if (currentChart) {
        currentChart.destroy();
        currentChart = null;
    }

    const labels = monthlyBalance.map(m => m.label);
    const debitData = monthlyBalance.map(m => m.debit);
    const creditData = monthlyBalance.map(m => m.credit);

    const isDark = document.body.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#94a3b8' : '#64748b';
    const gridColor = isDark ? '#334155' : '#e2e8f0';

    const ctx = canvas.getContext('2d');
    currentChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Incassi (Dare)',
                    data: debitData,
                    backgroundColor: '#10b981',
                    borderRadius: 6
                },
                {
                    label: 'Versamenti (Avere)',
                    data: creditData,
                    backgroundColor: '#0f2b5c',
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: textColor, font: { family: 'Inter', weight: '500' } }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) label += ': ';
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toLocaleString('it-IT', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + ' €';
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: textColor, font: { family: 'Inter' } },
                    grid: { color: gridColor }
                },
                y: {
                    ticks: {
                        color: textColor,
                        font: { family: 'Inter' },
                        callback: function(value) {
                            return value.toLocaleString('it-IT') + ' €';
                        }
                    },
                    grid: { color: gridColor }
                }
            }
        }
    });
}

export function updateAnomaliesTableUI() {
    const filterVal = document.getElementById('tableFilterSelect')?.value || 'all';
    const sortVal = document.getElementById('tableSortSelect')?.value || 'date_asc';

    let filtered = [...currentTableItems];
    if (filterVal !== 'all') {
        filtered = filtered.filter(item => item.type === filterVal);
    }

    filtered.sort((a, b) => {
        if (sortVal === 'date_asc') return a.dateVal - b.dateVal;
        if (sortVal === 'date_desc') return b.dateVal - a.dateVal;
        if (sortVal === 'amount_desc') return b.amount - a.amount;
        if (sortVal === 'amount_asc') return a.amount - b.amount;
        return 0;
    });

    const tbody = document.getElementById('anomaliesTableBody');
    const emptyState = document.getElementById('tableEmptyState');
    const tableCard = document.getElementById('anomaliesTable');

    if (!tbody) return;

    if (filtered.length === 0) {
        tbody.innerHTML = '';
        tableCard.classList.add('d-none');
        emptyState.classList.remove('d-none');
        return;
    }

    tableCard.classList.remove('d-none');
    emptyState.classList.add('d-none');

    const fmtEur = (v) => v.toLocaleString('it-IT', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + ' €';

    tbody.innerHTML = filtered.map(item => `
        <tr>
            <td><span class="badge ${item.badgeClass} rounded-pill px-2 py-1">${item.typeLabel}</span></td>
            <td><span class="fw-medium">${item.dateStr || '-'}</span></td>
            <td><code class="text-dark small fw-bold">${item.transactionId}</code></td>
            <td class="text-end fw-bold ${item.type === 'anomaly' ? 'text-danger' : 'text-body'}">
                ${fmtEur(item.amount)}
                ${item.diff > 0 && item.type === 'anomaly' ? `<div class="text-danger small" style="font-size: 0.75rem;">Scoperto: ${fmtEur(item.diff)}</div>` : ''}
            </td>
            <td class="text-muted small">${item.details}</td>
        </tr>
    `).join('');
}

export function renderAnomaliesTable(tableItems) {
    currentTableItems = tableItems || [];
    updateAnomaliesTableUI();
}
