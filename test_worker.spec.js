const { test, expect } = require('@playwright/test');
const path = require('path');
const fs = require('fs');

test('Standalone App: Web Worker processes Excel reconciliation', async ({ page }) => {
    const filePath = path.resolve(__dirname, 'app/cashrec.html');
    await page.goto(`file://${filePath}`);

    const fileContentBase64 = await page.evaluate(() => {
        const data = [
            { "Data Reg.": "01/01/2026", "Dare": 100.0, "Avere": 0, "Data Val.": "01/01/2026" },
            { "Data Reg.": "02/01/2026", "Dare": 50.0, "Avere": 0, "Data Val.": "02/01/2026" },
            { "Data Reg.": "02/01/2026", "Dare": 0, "Avere": 150.0, "Data Val.": "02/01/2026" }
        ];
        const ws = XLSX.utils.json_to_sheet(data);
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, "Sheet1");
        return XLSX.write(wb, { bookType: 'xlsx', type: 'base64' });
    });

    const testExcelPath = path.resolve(__dirname, 'test_sample.xlsx');
    fs.writeFileSync(testExcelPath, Buffer.from(fileContentBase64, 'base64'));

    try {
        const fileInput = await page.$('#fileInput');
        await fileInput.setInputFiles(testExcelPath);

        await page.click('#btnSubmit');

        await expect(page.locator('#resultContainer')).not.toHaveClass(/d-none/, { timeout: 15000 });

        const logText = await page.textContent('#logOutput');
        expect(logText).toContain('Report Excel generato con successo');

        await expect(page.locator('#matchBadgesSummary')).toBeVisible();
        await expect(page.locator('#chartCard')).toBeVisible();
        await expect(page.locator('#monthlyChart')).toBeVisible();
        await expect(page.locator('#anomaliesTableCard')).toBeVisible();
    } finally {
        if (fs.existsSync(testExcelPath)) {
            fs.unlinkSync(testExcelPath);
        }
    }
});

test('CSV file parsing and reconciliation works (Phase 9)', async ({ page }) => {
    const filePath = path.resolve(__dirname, 'app/cashrec.html');
    await page.goto(`file://${filePath}`);

    const csvContent = `Data Reg.;Dare;Avere;Data Val.
01/01/2026;100,00;0;01/01/2026
02/01/2026;50,00;0;02/01/2026
02/01/2026;0;150,00;02/01/2026`;

    const testCsvPath = path.resolve(__dirname, 'test_sample.csv');
    fs.writeFileSync(testCsvPath, csvContent, 'utf-8');

    try {
        const fileInput = await page.$('#fileInput');
        await fileInput.setInputFiles(testCsvPath);

        await page.click('#btnSubmit');
        await expect(page.locator('#resultContainer')).not.toHaveClass(/d-none/, { timeout: 15000 });

        const logText = await page.textContent('#logOutput');
        expect(logText).toContain('Report Excel generato con successo');
    } finally {
        if (fs.existsSync(testCsvPath)) {
            fs.unlinkSync(testCsvPath);
        }
    }
});

test('Theme toggle cycles through Light, Dark, High-Contrast and i18n switches language (Phases 11 & 13)', async ({ page }) => {
    const filePath = path.resolve(__dirname, 'app/cashrec.html');
    await page.goto(`file://${filePath}`);

    await page.click('#themeToggle');
    let theme = await page.getAttribute('body', 'data-theme');
    expect(theme).toBe('dark');

    await page.click('#themeToggle');
    theme = await page.getAttribute('body', 'data-theme');
    expect(theme).toBe('high-contrast');

    await page.click('#themeToggle');
    theme = await page.getAttribute('body', 'data-theme');
    expect(theme).toBe('light');

    await page.selectOption('#langSelect', 'en');
    const processBtnText = await page.textContent('[data-i18n="btn_process"]');
    expect(processBtnText).toContain('Process File');
});

test('IndexedDB history records run and opens history modal (Phase 7)', async ({ page }) => {
    const filePath = path.resolve(__dirname, 'app/cashrec.html');
    await page.goto(`file://${filePath}`);

    await page.click('#btnHistory');
    await expect(page.locator('#historyModal')).not.toHaveClass(/d-none/);

    await page.click('#historyModalClose');
    await expect(page.locator('#historyModal')).toHaveClass(/d-none/);
});

test('Phase 4 Dashboard table filtering and sorting work', async ({ page }) => {
    const filePath = path.resolve(__dirname, 'app/cashrec.html');
    await page.goto(`file://${filePath}`);

    const fileContentBase64 = await page.evaluate(() => {
        const data = [
            { "Data Reg.": "01/01/2026", "Dare": 10.0, "Avere": 0, "Data Val.": "01/01/2026" },
            { "Data Reg.": "05/01/2026", "Dare": 0, "Avere": 500.0, "Data Val.": "05/01/2026" },
            { "Data Reg.": "20/01/2026", "Dare": 250.0, "Avere": 0, "Data Val.": "20/01/2026" }
        ];
        const ws = XLSX.utils.json_to_sheet(data);
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, "Sheet1");
        return XLSX.write(wb, { bookType: 'xlsx', type: 'base64' });
    });

    const testExcelPath = path.resolve(__dirname, 'test_dashboard.xlsx');
    fs.writeFileSync(testExcelPath, Buffer.from(fileContentBase64, 'base64'));

    try {
        const fileInput = await page.$('#fileInput');
        await fileInput.setInputFiles(testExcelPath);

        await page.click('#btnSubmit');
        await expect(page.locator('#resultContainer')).not.toHaveClass(/d-none/, { timeout: 15000 });

        const rowCount = await page.locator('#anomaliesTableBody tr').count();
        expect(rowCount).toBeGreaterThan(0);

        await page.selectOption('#tableFilterSelect', 'anomaly');
        const anomalyRows = await page.locator('#anomaliesTableBody tr').count();
        expect(anomalyRows).toBeGreaterThanOrEqual(1);

        await page.selectOption('#tableSortSelect', 'amount_desc');
        await expect(page.locator('#anomaliesTableBody')).toBeVisible();
    } finally {
        if (fs.existsSync(testExcelPath)) {
            fs.unlinkSync(testExcelPath);
        }
    }
});

test('Web Worker cancellation via Annulla button works', async ({ page }) => {
    const filePath = path.resolve(__dirname, 'app/cashrec.html');
    await page.goto(`file://${filePath}`);

    const fileContentBase64 = await page.evaluate(() => {
        const data = Array.from({ length: 500 }, (_, i) => ({
            "Data Reg.": "01/01/2026",
            "Dare": i % 2 === 0 ? 10.0 : 0,
            "Avere": i % 2 === 1 ? 10.0 : 0
        }));
        const ws = XLSX.utils.json_to_sheet(data);
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, "Sheet1");
        return XLSX.write(wb, { bookType: 'xlsx', type: 'base64' });
    });

    const testExcelPath = path.resolve(__dirname, 'test_large_sample.xlsx');
    fs.writeFileSync(testExcelPath, Buffer.from(fileContentBase64, 'base64'));

    try {
        const fileInput = await page.$('#fileInput');
        await fileInput.setInputFiles(testExcelPath);

        await page.click('#btnSubmit');

        await page.click('#btnCancel');

        await expect(page.locator('#errorContainer')).not.toHaveClass(/d-none/);
        const errorText = await page.textContent('#errorMessage');
        expect(errorText).toContain("Elaborazione annullata dall'utente");
    } finally {
        if (fs.existsSync(testExcelPath)) {
            fs.unlinkSync(testExcelPath);
        }
    }
});
