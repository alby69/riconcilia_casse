# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: test_worker.spec.js >> Web Worker processes Excel reconciliation and updates progress UI
- Location: test_worker.spec.js:5:1

# Error details

```
Error: expect(locator).not.toHaveClass(expected) failed

Locator: locator('#resultContainer')
Expected pattern: not /d-none/
Received string: "mt-4 d-none text-center"
Timeout: 15000ms

Call log:
  - Expect "not toHaveClass" with timeout 15000ms
  - waiting for locator('#resultContainer')
    31 × locator resolved to <div id="resultContainer" class="mt-4 d-none text-center">…</div>
       - unexpected value "mt-4 d-none text-center"

```

```yaml
- navigation:
  - link "CashRec (Standalone)":
    - /url: "#"
    - img
    - text: CashRec (Standalone)
  - button ""
  - button ""
- img: CashRec Simplify your accounts, liberate your business.
- text: 
- heading "Trascina qui il tuo file Excel" [level=5]
- paragraph: o clicca per selezionare
- text:  Impostazioni Avanzate 
- button " Elabora File"
- text: v5.1 Standalone HTML App - Fully Client-Side
```

# Test source

```ts
  1  | const { test, expect } = require('@playwright/test');
  2  | const path = require('path');
  3  | const fs = require('fs');
  4  |
  5  | test('Web Worker processes Excel reconciliation and updates progress UI', async ({ page }) => {
  6  |     page.on('console', msg => console.log(`BROWSER CONSOLE (${msg.type()}): ${msg.text()}`));
  7  |     page.on('pageerror', err => console.log(`PAGE ERROR: ${err.message}`));
  8  |
  9  |     const filePath = path.resolve(__dirname, 'app/cashrec.html');
  10 |     await page.goto(`file://${filePath}`);
  11 |
  12 |     const fileContentBase64 = await page.evaluate(() => {
  13 |         const data = [
  14 |             { "Data Reg.": "01/01/2026", "Dare": 100.0, "Avere": 0, "Data Val.": "01/01/2026" },
  15 |             { "Data Reg.": "02/01/2026", "Dare": 50.0, "Avere": 0, "Data Val.": "02/01/2026" },
  16 |             { "Data Reg.": "02/01/2026", "Dare": 0, "Avere": 150.0, "Data Val.": "02/01/2026" }
  17 |         ];
  18 |         const ws = XLSX.utils.json_to_sheet(data);
  19 |         const wb = XLSX.utils.book_new();
  20 |         XLSX.utils.book_append_sheet(wb, ws, "Sheet1");
  21 |         return XLSX.write(wb, { bookType: 'xlsx', type: 'base64' });
  22 |     });
  23 |
  24 |     const testExcelPath = path.resolve(__dirname, 'test_sample.xlsx');
  25 |     fs.writeFileSync(testExcelPath, Buffer.from(fileContentBase64, 'base64'));
  26 |
  27 |     try {
  28 |         const fileInput = await page.$('#fileInput');
  29 |         await fileInput.setInputFiles(testExcelPath);
  30 |
  31 |         await page.click('#btnSubmit');
  32 |
  33 |         // Verify result container appears upon worker completion
> 34 |         await expect(page.locator('#resultContainer')).not.toHaveClass(/d-none/, { timeout: 15000 });
     |                                                            ^ Error: expect(locator).not.toHaveClass(expected) failed
  35 |
  36 |         // Verify log contains success message
  37 |         const logText = await page.textContent('#logOutput');
  38 |         expect(logText).toContain('Report Excel generato con successo');
  39 |     } finally {
  40 |         if (fs.existsSync(testExcelPath)) {
  41 |             fs.unlinkSync(testExcelPath);
  42 |         }
  43 |     }
  44 | });
  45 |
  46 | test('Web Worker cancellation via Annulla button works', async ({ page }) => {
  47 |     const filePath = path.resolve(__dirname, 'app/cashrec.html');
  48 |     await page.goto(`file://${filePath}`);
  49 |
  50 |     const fileContentBase64 = await page.evaluate(() => {
  51 |         const data = Array.from({ length: 500 }, (_, i) => ({
  52 |             "Data Reg.": "01/01/2026",
  53 |             "Dare": i % 2 === 0 ? 10.0 : 0,
  54 |             "Avere": i % 2 === 1 ? 10.0 : 0
  55 |         }));
  56 |         const ws = XLSX.utils.json_to_sheet(data);
  57 |         const wb = XLSX.utils.book_new();
  58 |         XLSX.utils.book_append_sheet(wb, ws, "Sheet1");
  59 |         return XLSX.write(wb, { bookType: 'xlsx', type: 'base64' });
  60 |     });
  61 |
  62 |     const testExcelPath = path.resolve(__dirname, 'test_large_sample.xlsx');
  63 |     fs.writeFileSync(testExcelPath, Buffer.from(fileContentBase64, 'base64'));
  64 |
  65 |     try {
  66 |         const fileInput = await page.$('#fileInput');
  67 |         await fileInput.setInputFiles(testExcelPath);
  68 |
  69 |         await page.click('#btnSubmit');
  70 |
  71 |         // Cancel process
  72 |         await page.click('#btnCancel');
  73 |
  74 |         // Error container should show cancellation
  75 |         await expect(page.locator('#errorContainer')).not.toHaveClass(/d-none/);
  76 |         const errorText = await page.textContent('#errorMessage');
  77 |         expect(errorText).toContain("Elaborazione annullata dall'utente");
  78 |     } finally {
  79 |         if (fs.existsSync(testExcelPath)) {
  80 |             fs.unlinkSync(testExcelPath);
  81 |         }
  82 |     }
  83 | });
  84 |
```