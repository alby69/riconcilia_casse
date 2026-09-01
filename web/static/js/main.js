import { DEFAULT_CONFIG, DEFAULT_PROFILES, I18N_TRANSLATIONS } from './config.js';
import { CashRecDB } from './history.js';
import { initPWA } from './pwa.js';
import {
    injectFavicon, showToast, customConfirm, customPrompt,
    renderKPIDashboard, renderMatchBadges, renderMonthlyChart,
    renderAnomaliesTable, updateAnomaliesTableUI
} from './ui.js';

injectFavicon();
initPWA();

let CONFIG = JSON.parse(localStorage.getItem('cashrec_config')) || DEFAULT_CONFIG;
let PROFILES = JSON.parse(localStorage.getItem('cashrec_profiles')) || DEFAULT_PROFILES;
let generatedReportBlob = null;
let generatedReportFilename = "report_riconciliazione.xlsx";
let fullLogText = "";
let currentWorker = null;

let selectedFilesQueue = [];
let currentSelectedFileIndex = 0;

function applyLanguage(lang) {
    const translations = I18N_TRANSLATIONS[lang] || I18N_TRANSLATIONS['it'];
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (translations[key]) {
            el.textContent = translations[key];
        }
    });
    localStorage.setItem('cashrec_lang', lang);
}

const langSelect = document.getElementById('langSelect');
if (langSelect) {
    const savedLang = localStorage.getItem('cashrec_lang') || 'it';
    langSelect.value = savedLang;
    applyLanguage(savedLang);
    langSelect.addEventListener('change', (e) => applyLanguage(e.target.value));
}

// Theme Logic
const themeToggle = document.getElementById('themeToggle');
const body = document.body;
const icon = themeToggle ? themeToggle.querySelector('i') : null;

let savedTheme = localStorage.getItem('theme');
if (!savedTheme) {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        savedTheme = 'dark';
    } else {
        savedTheme = 'light';
    }
}
body.setAttribute('data-theme', savedTheme);
if (icon) updateIcon(savedTheme);

if (themeToggle) {
    themeToggle.addEventListener('click', () => {
        const currentTheme = body.getAttribute('data-theme') || 'light';
        let newTheme = 'light';
        if (currentTheme === 'light') newTheme = 'dark';
        else if (currentTheme === 'dark') newTheme = 'high-contrast';
        else newTheme = 'light';

        body.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        if (icon) updateIcon(newTheme);
        showToast(`Tema impostato: ${newTheme.toUpperCase()}`, 'info');
    });
}

function updateIcon(theme) {
    if (!icon) return;
    if (theme === 'dark') icon.className = 'fas fa-sun';
    else if (theme === 'high-contrast') icon.className = 'fas fa-adjust';
    else icon.className = 'fas fa-moon';
}

// File Upload Area
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const filePreviewContainer = document.getElementById('filePreviewContainer');
const fileQueueList = document.getElementById('fileQueueList');

if (dropZone && fileInput) {
    dropZone.addEventListener('click', (e) => {
        if (!e.target.closest('.file-preview-card') && !e.target.closest('.btn-queue-select')) {
            fileInput.click();
        }
    });
    fileInput.addEventListener('change', handleFileSelect);

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });
    function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

    ['dragenter', 'dragover'].forEach(e => dropZone.addEventListener(e, () => dropZone.classList.add('dragover')));
    ['dragleave', 'drop'].forEach(e => dropZone.addEventListener(e, () => dropZone.classList.remove('dragover')));

    dropZone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect();
        }
    });
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function handleFileSelect() {
    if (fileInput.files.length > 0) {
        selectedFilesQueue = Array.from(fileInput.files);
        currentSelectedFileIndex = 0;
        renderFileQueueUI();
        filePreviewContainer.classList.remove('d-none');
        dropZone.style.borderColor = "var(--accent-color)";
    }
}

function renderFileQueueUI() {
    if (!fileQueueList) return;
    fileQueueList.innerHTML = selectedFilesQueue.map((file, idx) => {
        const isCsv = file.name.toLowerCase().endsWith('.csv');
        const iconClass = isCsv ? 'fa-file-csv text-info' : 'fa-file-excel text-success';
        const isActive = idx === currentSelectedFileIndex;
        return `
            <div class="file-preview-card ${isActive ? 'border-primary' : ''}">
                <i class="fas ${iconClass} fa-2x"></i>
                <div class="text-start">
                    <div class="fw-bold text-dark small text-truncate" style="max-width: 260px;">${file.name}</div>
                    <div class="text-muted text-xs" style="font-size: 0.75rem;">${formatFileSize(file.size)} ${isCsv ? '• CSV' : '• Excel'}</div>
                </div>
                ${selectedFilesQueue.length > 1 ? `
                    <button type="button" class="btn btn-xs ${isActive ? 'btn-primary' : 'btn-outline-secondary'} btn-queue-select rounded-pill ms-2" onclick="selectQueueIndex(${idx})">
                        ${isActive ? 'Selezionato' : 'Scegli'}
                    </button>
                ` : `
                    <span class="badge bg-success-subtle text-success border border-success-subtle ms-2 rounded-pill small">
                        <i class="fas fa-check-circle me-1"></i>Pronto
                    </span>
                `}
            </div>
        `;
    }).join('');
}

window.selectQueueIndex = function(idx) {
    currentSelectedFileIndex = idx;
    renderFileQueueUI();
    showToast(`Selezionato per l'elaborazione: ${selectedFilesQueue[idx].name}`, 'info');
};

window.toggleSettings = function() {
    const section = document.getElementById('advancedSettings');
    const icon = document.getElementById('toggleIcon');
    const isHidden = section.style.display === 'none' || section.style.display === '';
    section.style.display = isHidden ? 'block' : 'none';
    icon.className = isHidden ? 'fas fa-chevron-up ms-1' : 'fas fa-chevron-down ms-1';
};

window.toggleLog = function() {
    const log = document.getElementById('logOutput');
    log.style.display = (log.style.display === 'none' || log.style.display === '') ? 'block' : 'none';
};

// Profile Management
function renderProfileSelect() {
    const select = document.getElementById('profile_select');
    if (!select) return;
    select.innerHTML = '<option value="">-- Seleziona un profilo salvato --</option>';
    for (const name in PROFILES) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
    }
}

document.getElementById('profile_select')?.addEventListener('change', function() {
    const selected = this.value;
    if (selected && PROFILES[selected]) {
        populateForm(PROFILES[selected]);
    }
});

document.getElementById('btnSaveProfile')?.addEventListener('click', async function() {
    const name = await customPrompt("Inserisci il nome del nuovo profilo:", "", "Salva Profilo");
    if (!name || !name.trim()) return;

    const formEl = document.getElementById('uploadForm');
    const formData = new FormData(formEl);
    const params = {};
    for (const [key, value] of formData.entries()) {
        if (key !== 'file_input') params[key] = value;
    }

    PROFILES[name.trim()] = params;
    localStorage.setItem('cashrec_profiles', JSON.stringify(PROFILES));
    renderProfileSelect();
    document.getElementById('profile_select').value = name.trim();
    showToast("Profilo '" + name.trim() + "' salvato con successo!", 'success');
});

document.getElementById('btnDeleteProfile')?.addEventListener('click', async function() {
    const select = document.getElementById('profile_select');
    const selected = select.value;
    if (!selected) {
        showToast("Seleziona prima un profilo da eliminare.", 'warning');
        return;
    }
    if (!(await customConfirm("Sei sicuro di voler eliminare il profilo '" + selected + "'?", "Elimina Profilo"))) return;

    delete PROFILES[selected];
    localStorage.setItem('cashrec_profiles', JSON.stringify(PROFILES));
    renderProfileSelect();
    showToast("Profilo eliminato con successo.", 'success');
});

document.getElementById('btnSaveDefaultConfig')?.addEventListener('click', async function() {
    if (!(await customConfirm("Sei sicuro di voler salvare questi parametri come nuovi default?", "Salva Default"))) return;

    const formEl = document.getElementById('uploadForm');
    CONFIG.common.algorithm = formEl.elements['algorithm']?.value || CONFIG.common.algorithm;
    CONFIG.common.search_direction = formEl.elements['search_direction']?.value || CONFIG.common.search_direction;
    CONFIG.common.tolerance = parseFloat(formEl.elements['tolerance']?.value) || CONFIG.common.tolerance;
    CONFIG.common.days_window = parseInt(formEl.elements['days_window']?.value) || CONFIG.common.days_window;
    CONFIG.common.max_combinations = parseInt(formEl.elements['max_combinations']?.value) || CONFIG.common.max_combinations;
    CONFIG.common.residual_threshold = parseFloat(formEl.elements['residual_threshold']?.value) || CONFIG.common.residual_threshold;
    CONFIG.common.residual_days_window = parseInt(formEl.elements['residual_days_window']?.value) || CONFIG.common.residual_days_window;
    CONFIG.common.handover_days = parseInt(formEl.elements['handover_days']?.value) || CONFIG.common.handover_days;

    localStorage.setItem('cashrec_config', JSON.stringify(CONFIG));
    showToast("Configurazione predefinita salvata nel browser!", 'success');
});

function populateForm(params) {
    for (const key in params) {
        const element = document.querySelector(`[name="${key}"]`);
        if (element) {
            element.value = params[key];
        }
    }
}

function initializeForm() {
    populateForm(CONFIG.common);
    renderProfileSelect();
}

function showError(phase, message) {
    const errorContainer = document.getElementById('errorContainer');
    const errorPhase = document.getElementById('errorPhase');
    const errorMessage = document.getElementById('errorMessage');

    errorPhase.textContent = `Fase: ${phase}`;
    errorMessage.textContent = message;
    errorContainer.classList.remove('d-none');
}

function hideError() {
    const errorContainer = document.getElementById('errorContainer');
    errorContainer.classList.add('d-none');
}

function updateProgressUI(percent, statusText) {
    const progressBar = document.getElementById('progressBar');
    const statusTextEl = document.getElementById('progressStatusText');
    if (progressBar) {
        progressBar.style.width = `${percent}%`;
        progressBar.setAttribute('aria-valuenow', percent);
        progressBar.textContent = `${percent}%`;
    }
    if (statusTextEl && statusText) {
        statusTextEl.textContent = statusText;
    }
}

document.getElementById('tableFilterSelect')?.addEventListener('change', updateAnomaliesTableUI);
document.getElementById('tableSortSelect')?.addEventListener('change', updateAnomaliesTableUI);

document.getElementById('btnCancel')?.addEventListener('click', function() {
    if (currentWorker) {
        currentWorker.terminate();
        currentWorker = null;
    }
    document.getElementById('progressContainer').classList.add('d-none');
    document.getElementById('btnSubmit').disabled = false;
    showError("Annullamento", "Elaborazione annullata dall'utente.");
});

document.getElementById('uploadForm')?.addEventListener('submit', async function(e) {
    e.preventDefault();
    hideError();

    const file = (selectedFilesQueue.length > 0 && selectedFilesQueue[currentSelectedFileIndex])
        ? selectedFilesQueue[currentSelectedFileIndex]
        : (fileInput.files && fileInput.files[0]);

    if (!file) {
        showError("Selezione File", "Per favore, seleziona un file Excel o CSV prima di procedere.");
        return;
    }

    const btn = document.getElementById('btnSubmit');
    const progress = document.getElementById('progressContainer');
    const result = document.getElementById('resultContainer');
    const log = document.getElementById('logOutput');

    btn.disabled = true;
    progress.classList.remove('d-none');
    result.classList.add('d-none');
    log.style.display = 'none';
    log.textContent = '';
    fullLogText = "";

    updateProgressUI(0, `Avvio elaborazione di "${file.name}" nel Web Worker...`);
    const formData = new FormData(this);
    const columnMappingForm = {
        col_date: formData.get("col_date"),
        col_debit: formData.get("col_debit"),
        col_credit: formData.get("col_credit"),
        col_store_id: formData.get("col_store_id"),
        col_valuta_date: formData.get("col_valuta_date")
    };

    const engineParams = {
        tolerance: parseFloat(formData.get("tolerance")) || CONFIG.common.tolerance,
        days_window: parseInt(formData.get("days_window")) || CONFIG.common.days_window,
        max_combinations: parseInt(formData.get("max_combinations")) || CONFIG.common.max_combinations,
        residual_threshold: parseFloat(formData.get("residual_threshold")) || CONFIG.common.residual_threshold,
        residual_days_window: parseInt(formData.get("residual_days_window")) || CONFIG.common.residual_days_window,
        handover_days: parseInt(formData.get("handover_days")) || CONFIG.common.handover_days,
        search_direction: formData.get("search_direction") || CONFIG.common.search_direction,
        algorithm: formData.get("algorithm") || CONFIG.common.algorithm
    };

    try {
        const fileBuffer = await file.arrayBuffer();

        if (currentWorker) {
            currentWorker.terminate();
        }

        currentWorker = new Worker('/static/js/worker.js');

        currentWorker.onmessage = function(e) {
            const { type, percent, statusText, phase, message, reportBlob, reportFilename, fullLogText: workerLog, stats, dashboardData } = e.data;

            if (type === 'progress') {
                updateProgressUI(percent, statusText);
            } else if (type === 'success') {
                generatedReportBlob = reportBlob;
                generatedReportFilename = reportFilename;
                fullLogText = workerLog;
                log.textContent = fullLogText;

                renderKPIDashboard(stats);
                if (dashboardData) {
                    renderMatchBadges(dashboardData.matchCounts);
                    renderMonthlyChart(dashboardData.monthlyBalance);
                    renderAnomaliesTable(dashboardData.tableItems);
                }

                CashRecDB.saveRun({
                    fileName: file.name,
                    stats: stats,
                    dashboardData: dashboardData,
                    fullLogText: workerLog
                });

                progress.classList.add('d-none');
                btn.disabled = false;
                result.classList.remove('d-none');
                currentWorker = null;
            } else if (type === 'error') {
                progress.classList.add('d-none');
                btn.disabled = false;
                showError(phase || "Elaborazione", message);
                currentWorker = null;
            }
        };

        currentWorker.onerror = function(err) {
            progress.classList.add('d-none');
            btn.disabled = false;
            showError("Web Worker Error", err.message || "Errore sconosciuto nel Web Worker.");
            currentWorker = null;
        };

        currentWorker.postMessage({
            type: 'start',
            data: {
                fileBuffer: fileBuffer,
                fileName: file.name,
                columnMappingForm: columnMappingForm,
                engineParams: engineParams
            }
        }, [fileBuffer]);

    } catch (err) {
        progress.classList.add('d-none');
        btn.disabled = false;
        showError("Preparazione File", err.message);
        console.error(err);
    }
});

document.getElementById('btnDownload')?.addEventListener('click', function() {
    if (!generatedReportBlob) {
        showToast("Nessun report generato disponibile.", 'warning');
        return;
    }
    const url = URL.createObjectURL(generatedReportBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = generatedReportFilename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast("Download del Report Excel avviato!", 'success');
});

document.getElementById('btnDownloadPDF')?.addEventListener('click', async function() {
    const resultCard = document.querySelector('#resultContainer .alert-success');
    if (!resultCard || document.getElementById('resultContainer').classList.contains('d-none')) {
        showToast("Nessun risultato disponibile per l'export PDF.", 'warning');
        return;
    }

    showToast("Generazione del report PDF in corso...", 'info');

    const pdfContainer = document.createElement('div');
    pdfContainer.style.padding = '20px';
    pdfContainer.style.background = '#ffffff';
    pdfContainer.style.color = '#1e293b';
    pdfContainer.style.fontFamily = 'Inter, sans-serif';

    const chartCanvas = document.getElementById('monthlyChart');
    let chartImgHtml = '';
    if (chartCanvas) {
        const chartDataUrl = chartCanvas.toDataURL('image/png');
        chartImgHtml = `<div style="text-align: center; margin: 15px 0;"><img src="${chartDataUrl}" style="max-width: 100%; height: 200px;" /></div>`;
    }

    const kpiHtml = document.getElementById('kpiDashboard')?.innerHTML || '';
    const badgesHtml = document.getElementById('matchBadgesSummary')?.innerHTML || '';

    pdfContainer.innerHTML = `
        <div style="border-bottom: 2px solid #0f2b5c; padding-bottom: 10px; margin-bottom: 15px;">
            <h2 style="color: #0f2b5c; font-family: Outfit, sans-serif; font-weight: 800; margin: 0;">CashRec — Report Executive Riconciliazione</h2>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">Data Report: ${new Date().toLocaleString('it-IT')}</div>
        </div>
        <h4 style="font-size: 1.1rem; color: #0f2b5c; margin-bottom: 10px;">Indicatori Chiave di Prestazione (KPI)</h4>
        <div style="margin-bottom: 15px;">${kpiHtml}</div>
        <div style="margin-bottom: 15px;">${badgesHtml}</div>
        <h4 style="font-size: 1.1rem; color: #0f2b5c; margin-top: 20px; margin-bottom: 10px;">Quadratura Mensile</h4>
        ${chartImgHtml}
        <div style="margin-top: 20px; font-size: 0.75rem; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 8px;">
            Generato con CashRec Web App • 100% Client-Side
        </div>
    `;

    const opt = {
        margin:       10,
        filename:     `cashrec_report_${new Date().toISOString().slice(0, 10)}.pdf`,
        image:        { type: 'jpeg', quality: 0.98 },
        html2canvas:  { scale: 2, logging: false },
        jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' }
    };

    try {
        if (window.html2pdf) {
            await html2pdf().set(opt).from(pdfContainer).save();
            showToast("Download del Report PDF completato!", 'success');
        } else {
            window.print();
        }
    } catch (err) {
        console.warn("PDF generation error, fallback to print:", err);
        window.print();
    }
});

document.getElementById('btnDownloadLog')?.addEventListener('click', function() {
    if (!fullLogText) {
        showToast("Nessun log disponibile per il download.", 'warning');
        return;
    }
    const blob = new Blob([fullLogText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cashrec_log_${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast("Download del file Log avviato!", 'success');
});

document.getElementById('btnNewFile')?.addEventListener('click', function() {
    fileInput.value = '';
    filePreviewContainer.classList.add('d-none');
    dropZone.style.borderColor = '';

    hideError();
    document.getElementById('resultContainer').classList.add('d-none');
    document.getElementById('logOutput').style.display = 'none';
    document.getElementById('logOutput').textContent = '';

    document.getElementById('uploadForm').reset();
    populateForm(CONFIG.common);
});

// Help & History Modals
(function() {
    const modal = document.getElementById('helpModal');
    if (!modal) return;
    const close = () => modal.classList.add('d-none');
    const open = () => modal.classList.remove('d-none');
    document.getElementById('btnHelp')?.addEventListener('click', open);
    document.getElementById('helpModalClose')?.addEventListener('click', close);
    modal.addEventListener('click', (e) => { if (e.target === modal) close(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
})();

(function() {
    const modal = document.getElementById('historyModal');
    const listContainer = document.getElementById('historyListContainer');
    if (!modal || !listContainer) return;

    const open = async () => {
        const history = await CashRecDB.getHistory();
        if (history.length === 0) {
            listContainer.innerHTML = '<div class="text-center text-muted py-4 small"><i class="fas fa-folder-open fa-2x mb-2"></i><br>Nessuna elaborazione salvata in cronologia.</div>';
        } else {
            listContainer.innerHTML = history.map(item => `
                <div class="list-group-item list-group-item-action d-flex align-items-center justify-content-between gap-3 p-3" style="background: var(--bg-color); border-color: var(--border-color);">
                    <div>
                        <div class="fw-bold small text-dark"><i class="fas fa-file-excel text-success me-2"></i>${item.fileName}</div>
                        <div class="text-muted text-xs" style="font-size: 0.78rem;"><i class="fas fa-clock me-1"></i>${item.dateStr}</div>
                    </div>
                    <div class="d-flex align-items-center gap-2">
                        <span class="badge bg-primary-subtle text-primary border border-primary-subtle rounded-pill small">
                            ${item.stats._raw_debit_amount_perc.toFixed(1)}% Copertura
                        </span>
                        <button type="button" class="btn btn-sm btn-outline-primary rounded-pill btn-restore-history" data-timestamp="${item.timestamp}">
                            <i class="fas fa-folder-open me-1"></i>Riapri
                        </button>
                    </div>
                </div>
            `).join('');

            listContainer.querySelectorAll('.btn-restore-history').forEach(btn => {
                btn.addEventListener('click', () => {
                    const ts = parseInt(btn.getAttribute('data-timestamp'));
                    const record = history.find(h => h.timestamp === ts);
                    if (record) {
                        renderKPIDashboard(record.stats);
                        if (record.dashboardData) {
                            renderMatchBadges(record.dashboardData.matchCounts);
                            renderMonthlyChart(record.dashboardData.monthlyBalance);
                            renderAnomaliesTable(record.dashboardData.tableItems);
                        }
                        fullLogText = record.fullLogText || "";
                        document.getElementById('logOutput').textContent = fullLogText;
                        document.getElementById('resultContainer').classList.remove('d-none');
                        modal.classList.add('d-none');
                        showToast("Riepilogo caricato dalla cronologia!", "info");
                    }
                });
            });
        }
        modal.classList.remove('d-none');
    };

    const close = () => modal.classList.add('d-none');

    document.getElementById('btnHistory')?.addEventListener('click', open);
    document.getElementById('historyModalClose')?.addEventListener('click', close);
    document.getElementById('btnClearHistory')?.addEventListener('click', async () => {
        if (await customConfirm("Sei sicuro di voler svuotare l'intera cronologia?", "Svuota Cronologia")) {
            await CashRecDB.clearHistory();
            open();
            showToast("Cronologia svuotata.", "success");
        }
    });
    modal.addEventListener('click', (e) => { if (e.target === modal) close(); });
})();

document.addEventListener('DOMContentLoaded', initializeForm);
