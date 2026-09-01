export function robustCurrencyParser(value) {
    if (typeof value === 'number') return value;
    if (value === null || value === undefined) return 0;
    let str = String(value).trim().replace(/€/g, '').replace(/\s+/g, '');
    if (str === '') return 0;
    if (str.includes('.') && str.includes(',')) {
        str = str.replace(/\./g, '').replace(',', '.');
    } else if (str.includes(',')) {
        str = str.replace(',', '.');
    }
    const parsed = parseFloat(str);
    return isNaN(parsed) ? 0 : parsed;
}

export function parseItalianDate(val) {
    if (!val) return null;
    if (val instanceof Date) return isNaN(val.getTime()) ? null : val;

    if (typeof val === 'number' && typeof XLSX !== 'undefined') {
        const parsed = XLSX.SSF.parse_date_code(val);
        if (parsed) return new Date(Date.UTC(parsed.y, parsed.m - 1, parsed.d, parsed.H, parsed.M, parsed.S));
    }

    const str = String(val).trim();
    const euMatch = str.match(/^(\d{1,2})[\/\.-](\d{1,2})[\/\.-](\d{4})/);
    if (euMatch) {
        const day = parseInt(euMatch[1], 10);
        const month = parseInt(euMatch[2], 10) - 1;
        const year = parseInt(euMatch[3], 10);
        return new Date(Date.UTC(year, month, day));
    }
    const d = new Date(str);
    return isNaN(d.getTime()) ? null : d;
}

export function parseCSVText(csvText) {
    const lines = csvText.split(/\r?\n/);
    if (lines.length === 0) return [];

    const firstLine = lines[0] || "";
    const semicoCount = (firstLine.match(/;/g) || []).length;
    const commaCount = (firstLine.match(/,/g) || []).length;
    const delimiter = semicoCount >= commaCount ? ';' : ',';

    const parseLine = (line) => {
        const result = [];
        let current = '';
        let inQuotes = false;
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            if (char === '"') {
                if (inQuotes && line[i+1] === '"') {
                    current += '"';
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (char === delimiter && !inQuotes) {
                result.push(current.trim());
                current = '';
            } else {
                current += char;
            }
        }
        result.push(current.trim());
        return result;
    };

    const headers = parseLine(lines[0]);
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        const vals = parseLine(line);
        const row = {};
        headers.forEach((h, idx) => {
            row[h] = vals[idx] !== undefined ? vals[idx] : "";
        });
        rows.push(row);
    }
    return rows;
}

export async function readExcelFile(file, fileName, columnMappingForm, logMessages = [], progressCallback = null) {
    const buffer = await file.arrayBuffer();
    let rawRows = [];

    if (fileName && fileName.toLowerCase().endsWith('.csv')) {
        const textDecoder = new TextDecoder('utf-8');
        const csvText = textDecoder.decode(buffer);
        rawRows = parseCSVText(csvText);
    } else {
        const workbook = XLSX.read(buffer, { type: 'array', cellDates: true });
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        rawRows = XLSX.utils.sheet_to_json(worksheet, { defval: "" });
    }

    if (rawRows.length === 0) {
        throw new Error("Il file caricato appare vuoto.");
    }

    const MAX_ROWS = 50000;
    if (rawRows.length > MAX_ROWS) {
        throw new Error("Il file contiene " + rawRows.length.toLocaleString('it-IT') + " righe, superando il limite massimo consentito di " + MAX_ROWS.toLocaleString('it-IT') + " righe.");
    }

    const headerKeys = Object.keys(rawRows[0]);

    const findCol = (...candidates) => {
        for (const name of candidates) {
            if (!name || typeof name !== 'string' || !name.trim()) continue;
            const match = headerKeys.find(k => k.trim().toLowerCase() === name.trim().toLowerCase());
            if (match) return match;
        }
        return null;
    };

    const colDateInput = columnMappingForm.col_date?.trim();
    const colDebitInput = columnMappingForm.col_debit?.trim();
    const colCreditInput = columnMappingForm.col_credit?.trim();
    const colStoreIdInput = columnMappingForm.col_store_id?.trim();
    const colValutaDateInput = columnMappingForm.col_valuta_date?.trim();

    const actualColDate = findCol(colDateInput, "Data Reg.", "Data", "Date");
    const actualColDebit = findCol(colDebitInput, "Dare", "Debit");
    const actualColCredit = findCol(colCreditInput, "Avere", "Credit");
    const actualColStoreId = findCol(colStoreIdInput, "Codice Negozio", "Store", "Store ID");
    const actualColValutaDate = findCol(colValutaDateInput, "Data Val.", "Data Valuta", "Valuta");

    const missingCols = [];
    if (!actualColDate) missingCols.push("Data ('" + (colDateInput || 'Data Reg.') + "')");
    if (!actualColDebit) missingCols.push("Dare ('" + (colDebitInput || 'Dare') + "')");
    if (!actualColCredit) missingCols.push("Avere ('" + (colCreditInput || 'Avere') + "')");

    if (missingCols.length > 0) {
        throw new Error("Colonne obbligatorie non trovate nel file Excel: " + missingCols.join(', ') + ". Verifica i nomi delle colonne nelle Impostazioni Avanzate.");
    }

    const rows = [];
    let skippedInvalidDate = 0;
    let skippedEmptyAmounts = 0;

    for (let i = 0; i < rawRows.length; i++) {
        const raw = rawRows[i];

        const rawDate = raw[actualColDate] !== undefined ? raw[actualColDate] : raw["Date"];
        const parsedDate = parseItalianDate(rawDate);
        if (!parsedDate) {
            skippedInvalidDate++;
            continue;
        }

        const debitVal = robustCurrencyParser(raw[actualColDebit] !== undefined ? raw[actualColDebit] : raw["Debit"]);
        const creditVal = robustCurrencyParser(raw[actualColCredit] !== undefined ? raw[actualColCredit] : raw["Credit"]);

        if (debitVal === 0 && creditVal === 0) {
            skippedEmptyAmounts++;
        }

        const debitCents = Math.round(debitVal * 100);
        const creditCents = Math.round(creditVal * 100);

        const rowObj = {
            orig_index: i,
            Date: parsedDate,
            Debit: debitCents,
            Credit: creditCents,
            store_id: actualColStoreId && raw[actualColStoreId] !== undefined ? raw[actualColStoreId] : null
        };

        if (actualColValutaDate && raw[actualColValutaDate] !== undefined && raw[actualColValutaDate] !== "") {
            const parsedValuta = parseItalianDate(raw[actualColValutaDate]);
            if (parsedValuta) {
                rowObj.valuta_date = parsedValuta;
            }
        }

        if (raw["Saldo Prog."] !== undefined && raw["Saldo Prog."] !== "") {
            rowObj["Saldo Prog."] = robustCurrencyParser(raw["Saldo Prog."]);
        }

        rows.push(rowObj);
    }

    if (progressCallback) progressCallback(15, "Lettura completata: " + rawRows.length + " record trovati.");
    if (logMessages) {
        logMessages.push("[Parsing] Letti " + rawRows.length + " record dal file Excel.");
        if (skippedInvalidDate > 0) {
            logMessages.push(`[Parsing] Avviso: ${skippedInvalidDate} righe scartate perché senza data valida.`);
        }
        if (skippedEmptyAmounts > 0) {
            logMessages.push(`[Parsing] Info: ${skippedEmptyAmounts} righe con importi Dare/Avere pari a zero.`);
        }
        logMessages.push(`[Parsing] ${rows.length} righe valide pronte per la riconciliazione.`);
    }

    if (rows.length === 0) {
        throw new Error("Nessuna riga valida con data trovata nel file Excel.");
    }

    return rows;
}
