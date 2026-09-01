const { test, expect } = require('@playwright/test');
const path = require('path');
const fs = require('fs');

test('Web Worker processes Excel reconciliation and updates progress UI', async ({ page }) => {
    page.on('console', msg => console.log(`BROWSER CONSOLE (${msg.type()}): ${msg.text()}`));
    page.on('pageerror', err => console.log(`PAGE ERROR: ${err.message}`));

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

        // Verify result container appears upon worker completion
        await expect(page.locator('#resultContainer')).not.toHaveClass(/d-none/, { timeout: 15000 });

        // Verify log contains success message
        const logText = await page.textContent('#logOutput');
        expect(logText).toContain('Report Excel generato con successo');

        // Verify Phase 4 Dashboard UI elements (Badges, Chart, Table)
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

test('Phase 4 Dashboard table filtering and sorting work', async ({ page }) => {
    const filePath = path.resolve(__dirname, 'app/cashrec.html');
    await page.goto(`file://${filePath}`);

    const fileContentBase64 = await page.evaluate(() => {
        const data = [
            // Sample data that causes an anomaly, an unused debit, and an unreconciled credit
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

        // Table should have rows
        const rowCount = await page.locator('#anomaliesTableBody tr').count();
        expect(rowCount).toBeGreaterThan(0);

        // Change filter to anomaly
        await page.selectOption('#tableFilterSelect', 'anomaly');
        const anomalyRows = await page.locator('#anomaliesTableBody tr').count();
        expect(anomalyRows).toBeGreaterThanOrEqual(1);

        // Change sort option to amount_desc
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

        // Cancel process
        await page.click('#btnCancel');

        // Error container should show cancellation
        await expect(page.locator('#errorContainer')).not.toHaveClass(/d-none/);
        const errorText = await page.textContent('#errorMessage');
        expect(errorText).toContain("Elaborazione annullata dall'utente");
    } finally {
        if (fs.existsSync(testExcelPath)) {
            fs.unlinkSync(testExcelPath);
        }
    }
});
