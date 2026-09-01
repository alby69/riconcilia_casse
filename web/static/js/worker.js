importScripts('https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js');
importScripts('https://cdn.jsdelivr.net/npm/exceljs@4.4.0/dist/exceljs.min.js');

function robustCurrencyParser(value) {
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

function parseItalianDate(val) {
    if (!val) return null;
    if (val instanceof Date) return isNaN(val.getTime()) ? null : val;

    if (typeof val === 'number') {
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

function parseCSVText(csvText) {
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

async function readExcelFile(file, fileName, columnMappingForm, logMessages = [], progressCallback = null) {
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

class JSReconciliationEngine {
    constructor(params) {
        this.progressCallback = params.progressCallback || null;
        this.tolerance = Math.round((params.tolerance || 50.0) * 100);
        this.days_window = params.days_window || 5;
        this.max_combinations = params.max_combinations || 10;
        this.residual_threshold = Math.round((params.residual_threshold || 50.0) * 100);
        this.residual_days_window = params.residual_days_window || 5;
        this.search_direction = params.search_direction || "past_only";
        this.algorithm = params.algorithm || "progressive_balance";
        this.sorting_strategy = params.sorting_strategy || "date";
        this.ignore_tolerance = params.ignore_tolerance || false;
        this.enable_best_fit = params.enable_best_fit !== undefined ? params.enable_best_fit : true;
        this.handover_days = params.handover_days || 5;

        this.used_debit_indices = new Set();
        this.used_credit_indices = new Set();
        this.max_id_counter = 0;

        this.debit_df = [];
        this.credit_df = [];
        this.matches = [];
    }

    calculateTimeWindow(refDate, daysWindow, searchDir) {
        const minDate = new Date(refDate);
        const maxDate = new Date(refDate);

        if (searchDir === "future_only") {
            maxDate.setDate(maxDate.getDate() + daysWindow);
        } else if (searchDir === "past_only") {
            minDate.setDate(minDate.getDate() - daysWindow);
        } else {
            minDate.setDate(minDate.getDate() - daysWindow);
            maxDate.setDate(maxDate.getDate() + daysWindow);
        }
        return { minDate, maxDate };
    }

    separateMovements(rows) {
        this.max_id_counter = rows.reduce((max, r) => Math.max(max, r.orig_index), 0);

        this.debit_df = rows.filter(r => r.Debit !== 0).map(r => ({ ...r }));
        this.credit_df = rows.filter(r => r.Credit !== 0).map(r => ({
            ...r,
            effective_date: r.valuta_date || r.Date,
            analysis_date: r.valuta_date || r.Date
        }));

        this.credit_df.forEach(r => {
            if (r.valuta_date && r.Date && r.valuta_date.getFullYear() !== r.Date.getFullYear()) {
                r.effective_date = r.Date;
                r.analysis_date = r.Date;
            }
        });

        this.debit_df.sort((a, b) => a.Date - b.Date);
        this.credit_df.sort((a, b) => a.analysis_date - b.analysis_date);
    }

    static monthKey(date) {
        return `${date.getFullYear()}-${date.getMonth()}`;
    }

    static monthKeyPrev(key) {
        const [y, m] = key.split("-").map(Number);
        const d = new Date(y, m - 1, 1);
        return JSReconciliationEngine.monthKey(d);
    }

    static periodCompare(a, b) {
        const [ay, am] = a.split("-").map(Number);
        const [by, bm] = b.split("-").map(Number);
        if (ay !== by) return ay - by;
        return am - bm;
    }

    creditMatchesPreviousMonth(creditOrigIdx, prevKey) {
        if (!this.matches.length) return false;
        for (const row of this.matches) {
            if (!(row.credit_indices || []).includes(creditOrigIdx)) continue;
            const debitDates = row.debit_dates || [];
            if (!debitDates.length) return false;
            const debitMonths = new Set(debitDates.map(d => JSReconciliationEngine.monthKey(d)));
            return debitMonths.size === 1 && debitMonths.has(prevKey);
        }
        return false;
    }

    economicMonthOfCredit(creditRow, minKey, maxKey) {
        const regKey = JSReconciliationEngine.monthKey(creditRow.Date);
        const prevKey = JSReconciliationEngine.monthKeyPrev(regKey);

        let carryBack = this.handover_days > 0
            && creditRow.Date.getDate() <= this.handover_days
            && this.creditMatchesPreviousMonth(creditRow.orig_index, prevKey);
        if (carryBack && creditRow.valuta_date) {
            if (JSReconciliationEngine.monthKey(creditRow.valuta_date) === regKey) {
                carryBack = false;
            }
        }

        let period;
        if (carryBack) {
            period = prevKey;
        } else {
            period = JSReconciliationEngine.monthKey(creditRow.valuta_date || creditRow.Date);
        }

        if (JSReconciliationEngine.periodCompare(period, minKey) < 0) period = minKey;
        if (JSReconciliationEngine.periodCompare(period, maxKey) > 0) period = maxKey;
        return period;
    }

    computeMonthlyTotals() {
        const totals = {};
        if (!this.debit_df.length && !this.credit_df.length) return totals;

        const all = this.debit_df.concat(this.credit_df);
        let minKey = null, maxKey = null;
        all.forEach(r => {
            const k = JSReconciliationEngine.monthKey(r.Date);
            if (minKey === null || JSReconciliationEngine.periodCompare(k, minKey) < 0) minKey = k;
            if (maxKey === null || JSReconciliationEngine.periodCompare(k, maxKey) > 0) maxKey = k;
        });

        this.debit_df.forEach(r => {
            const k = JSReconciliationEngine.monthKey(r.Date);
            if (!totals[k]) totals[k] = { Debit: 0, Credit: 0 };
            totals[k].Debit += r.Debit;
        });
        this.credit_df.forEach(r => {
            const k = this.economicMonthOfCredit(r, minKey, maxKey);
            if (!totals[k]) totals[k] = { Debit: 0, Credit: 0 };
            totals[k].Credit += r.Credit;
        });
        return totals;
    }

    registerMatch(match) {
        if (!match) return;
        (match.debit_indices || []).forEach(i => this.used_debit_indices.add(i));
        (match.credit_indices || []).forEach(i => this.used_credit_indices.add(i));

        const creditDates = match.credit_dates || [];
        const minCreditDate = creditDates.length > 0 ? new Date(Math.min(...creditDates.map(d => d.getTime()))) : null;

        const tId = `D(${(match.debit_indices || []).map(i => i + 2).join(",")})_A(${(match.credit_indices || []).map(i => i + 2).join(",")})`;

        this.matches.push({
            "Transaction ID": tId,
            debit_indices: match.debit_indices || [],
            debit_dates: match.debit_dates || [],
            debit_amounts: match.debit_amounts || [],
            total_debit: match.total_debit || 0,
            credit_date: minCreditDate,
            num_credits: (match.credit_indices || []).length,
            credit_indices: match.credit_indices || [],
            credit_amounts: match.credit_amounts || [],
            total_credit: match.total_credit || match.total_debit || 0,
            difference: match.difference || 0,
            match_type: match.match_type || "N/D",
            pass_name: match.pass_name || "N/D",
            is_forced: !!match.is_forced
        });
    }

    reconcileProgressiveBalance() {
        const unusedDebits = this.debit_df.filter(r => !this.used_debit_indices.has(r.orig_index));
        const unusedCredits = this.credit_df.filter(r => !this.used_credit_indices.has(r.orig_index));

        const debitRemaining = {};
        unusedDebits.forEach((r, idx) => { debitRemaining[idx] = r.Debit; });

        for (let cIdx = 0; cIdx < unusedCredits.length; cIdx++) {
            const creditRow = unusedCredits[cIdx];
            const creditAmount = creditRow.Credit;
            const creditOrigIdx = creditRow.orig_index;
            const creditDate = creditRow.analysis_date;

            const { minDate, maxDate } = this.calculateTimeWindow(creditDate, this.days_window, this.search_direction);

            const candidateDebitIndices = [];

            for (let dIdx = 0; dIdx < unusedDebits.length; dIdx++) {
                if (debitRemaining[dIdx] > 0) {
                    const dDate = unusedDebits[dIdx].Date;
                    if (dDate >= minDate && dDate <= maxDate) {
                        candidateDebitIndices.push(dIdx);
                    }
                }
            }

            let sameMonth = false;
            if (candidateDebitIndices.length > 0) {
                const firstCandDate = unusedDebits[candidateDebitIndices[0]].Date;
                sameMonth = (creditDate.getFullYear() === firstCandDate.getFullYear() && creditDate.getMonth() === firstCandDate.getMonth());
            }

            if (!sameMonth && candidateDebitIndices.length > 0) {
                const firstCandDate = unusedDebits[candidateDebitIndices[0]].Date;
                if (creditDate.getFullYear() < firstCandDate.getFullYear() ||
                   (creditDate.getFullYear() === firstCandDate.getFullYear() && creditDate.getMonth() < firstCandDate.getMonth())) {
                    this.registerMatch({
                        debit_indices: [], debit_dates: [], debit_amounts: [], total_debit: 0,
                        credit_indices: [creditOrigIdx], credit_dates: [creditDate], credit_amounts: [creditAmount], total_credit: creditAmount,
                        difference: creditAmount,
                        match_type: `VERSAMENTO MESE PRECEDENTE: ${(creditAmount / 100).toFixed(2)}€ (non agganciato - periodo precedente)`,
                        pass_name: "Progressive Balance", is_forced: true
                    });
                    continue;
                }
            }

            if (candidateDebitIndices.length === 0) {
                this.registerMatch({
                    debit_indices: [], debit_dates: [], debit_amounts: [], total_debit: 0,
                    credit_indices: [creditOrigIdx], credit_dates: [creditDate], credit_amounts: [creditAmount], total_credit: creditAmount,
                    difference: creditAmount,
                    match_type: `VERSAMENTO SENZA INCASSI: ${(creditAmount / 100).toFixed(2)}€ (mese/anno successivo o senza dati)`,
                    pass_name: "Progressive Balance", is_forced: true
                });
                continue;
            }

            const currentMatchDebits = [];
            const currentDebitAmounts = [];
            let remainingCredit = creditAmount;

            for (const dIdx of candidateDebitIndices) {
                if (remainingCredit <= 0) break;
                const dAmount = debitRemaining[dIdx];
                const dOrigIdx = unusedDebits[dIdx].orig_index;

                if (dAmount <= remainingCredit) {
                    currentMatchDebits.push(dOrigIdx);
                    currentDebitAmounts.push(dAmount);
                    remainingCredit -= dAmount;
                    debitRemaining[dIdx] = 0;
                } else {
                    currentMatchDebits.push(dOrigIdx);
                    currentDebitAmounts.push(remainingCredit);
                    debitRemaining[dIdx] = dAmount - remainingCredit;
                    remainingCredit = 0;
                }
            }

            const totalDebitUsed = currentDebitAmounts.reduce((a, b) => a + b, 0);
            const difference = creditAmount - totalDebitUsed;
            const absDiff = Math.abs(difference);

            let matchObj = {
                debit_indices: [...currentMatchDebits],
                debit_dates: candidateDebitIndices.filter(dIdx => currentMatchDebits.includes(unusedDebits[dIdx].orig_index)).map(dIdx => unusedDebits[dIdx].Date),
                debit_amounts: [...currentDebitAmounts],
                total_debit: totalDebitUsed,
                credit_indices: [creditOrigIdx],
                credit_dates: [creditDate],
                credit_amounts: [creditAmount],
                total_credit: creditAmount,
                difference: absDiff,
                pass_name: "Progressive Balance"
            };

            if (absDiff <= this.tolerance && difference > 0) {
                matchObj.match_type = `Match: ${currentMatchDebits.length}D vs 1C (eccedenza versamento: +${(difference / 100).toFixed(2)}€)`;
            } else if (difference === 0) {
                matchObj.match_type = `Match: ${currentMatchDebits.length}D vs 1C`;
            } else if (difference > 0) {
                matchObj.match_type = `ANOMALY: ${(difference / 100).toFixed(2)}€ non coperti (differenza oltre tolleranza)`;
                matchObj.is_forced = true;
            } else {
                matchObj.match_type = `Match: ${currentMatchDebits.length}D vs 1C (eccedenza incasso: ${(difference / 100).toFixed(2)}€)`;
            }

            this.registerMatch(matchObj);
        }
    }

    reconcileSubsetSum() {
        if (this.progressCallback) this.progressCallback(35, "Subset Sum Pass 1: Aggregazione incassi...");
        this.runGenericPass(
            this.credit_df.filter(c => !this.used_credit_indices.has(c.orig_index)),
            this.debit_df,
            "Credit", "Debit", this.used_debit_indices,
            this.days_window, this.max_combinations,
            "Pass 1: Receipt Aggregation (Many DEBIT -> 1 CREDIT)",
            this.search_direction, this.findDebitMatches.bind(this), true
        );

        if (this.progressCallback) this.progressCallback(50, "Subset Sum Pass 2: Scomposizione versamenti...");
        let pass2Dir = this.search_direction === "past_only" ? "future_only" : (this.search_direction === "future_only" ? "past_only" : "both");
        this.runGenericPass(
            this.debit_df.filter(d => !this.used_debit_indices.has(d.orig_index)),
            this.credit_df,
            "Debit", "Credit", this.used_credit_indices,
            this.days_window, this.max_combinations,
            "Pass 2: Split Deposits (1 DEBIT -> Many CREDIT)",
            pass2Dir, this.findMatches.bind(this), false
        );

        if (this.progressCallback) this.progressCallback(65, "Subset Sum Pass 3: Recupero residui...");
        this.runGenericPass(
            this.credit_df.filter(c => !this.used_credit_indices.has(c.orig_index)),
            this.debit_df,
            "Credit", "Debit", this.used_debit_indices,
            this.residual_days_window, this.max_combinations,
            "Pass 3: Residual Recovery (Extended window: " + this.residual_days_window + "d)",
            this.search_direction, this.findDebitMatches.bind(this), false
        );
    }

    reconcileGreedyAmountFirst() {
        const debits = this.debit_df.filter(d => !this.used_debit_indices.has(d.orig_index)).sort((a, b) => b.Debit - a.Debit);
        const credits = this.credit_df.filter(c => !this.used_credit_indices.has(c.orig_index)).sort((a, b) => b.Credit - a.Credit);

        if (debits.length > credits.length) {
            this.runGenericPass(debits, credits, "Debit", "Credit", this.used_credit_indices, this.days_window, this.max_combinations, "Greedy Pass (Debit -> Credit)", this.search_direction, this.findMatches.bind(this), true);
        } else {
            this.runGenericPass(credits, debits, "Credit", "Debit", this.used_debit_indices, this.days_window, this.max_combinations, "Greedy Pass (Credit -> Debit)", this.search_direction, this.findDebitMatches.bind(this), true);
        }
    }

    runGenericPass(dfToProcess, dfCandidates, colToProcess, colCandidates, usedIndicesCandidates, daysWindow, maxCombinations, title, searchDirection, findFunction, enableBestFit) {
        for (const recordRow of dfToProcess) {
            const refDate = (colToProcess === "Credit" && recordRow.effective_date) ? recordRow.effective_date : recordRow.Date;
            let effectiveDir = searchDirection;
            if (colToProcess === "Credit" && recordRow.valuta_date) effectiveDir = "both";

            const { minDate, maxDate } = this.calculateTimeWindow(refDate, daysWindow, effectiveDir);

            const candidatesPrefiltered = dfCandidates.filter(c => {
                if (usedIndicesCandidates.has(c.orig_index)) return false;
                const candDate = c.Date;
                if (colToProcess === "Credit" && recordRow.effective_date) {
                    const regDate = recordRow.Date;
                    let filterDate = recordRow.effective_date;
                    if (regDate && filterDate.getFullYear() !== regDate.getFullYear()) {
                        filterDate = regDate;
                    }
                    if (candDate.getFullYear() > filterDate.getFullYear()) return false;
                    if (candDate.getFullYear() === filterDate.getFullYear() && candDate.getMonth() > filterDate.getMonth()) return false;
                } else if (colToProcess === "Debit") {
                    const creditValuta = c.valuta_date;
                    const creditEffective = c.effective_date || c.Date;
                    const filterDate = (creditValuta && creditValuta instanceof Date && !isNaN(creditValuta)) ? creditValuta : creditEffective;
                    if (filterDate.getFullYear() > recordRow.Date.getFullYear()) return false;
                    if (filterDate.getFullYear() === recordRow.Date.getFullYear() && filterDate.getMonth() > recordRow.Date.getMonth()) return false;
                }
                return candDate >= minDate && candDate <= maxDate;
            });

            if (candidatesPrefiltered.length > 0) {
                const match = findFunction(recordRow, candidatesPrefiltered, daysWindow, maxCombinations, enableBestFit);
                if (match) {
                    match.pass_name = title;
                    this.registerMatch(match);
                }
            }
        }
    }

    findMatches(debitRow, creditCandidates) {
        const debitAmount = debitRow.Debit;
        const debitDate = debitRow.Date;

        const candidates = creditCandidates.filter(c => c.Credit <= debitAmount + this.tolerance);
        if (candidates.length === 0) return null;

        const exactMatches = candidates.filter(c => Math.abs(c.Credit - debitAmount) <= this.tolerance);
        if (exactMatches.length > 0) {
            exactMatches.sort((a, b) => Math.abs(debitDate - a.Date) - Math.abs(debitDate - b.Date));
            const best = exactMatches[0];
            return {
                debit_indices: [debitRow.orig_index], debit_dates: [debitDate], debit_amounts: [debitAmount],
                credit_indices: [best.orig_index], credit_dates: [best.Date], credit_amounts: [best.Credit],
                total_credit: best.Credit, difference: Math.abs(debitAmount - best.Credit), match_type: "1-to-1"
            };
        }
        return null;
    }

    findDebitMatches(creditRow, debitCandidates) {
        const creditAmount = creditRow.Credit;
        const creditDate = creditRow.Date;

        const candidates = debitCandidates.filter(d => d.Debit <= creditAmount + this.tolerance);
        if (candidates.length === 0) return null;

        const match = this.findCombinationRecursive(creditAmount, candidates, this.max_combinations, this.tolerance);
        if (match) {
            const totalDebit = match.reduce((sum, m) => sum + m.Debit, 0);
            const difference = creditAmount >= totalDebit ? creditAmount - totalDebit : totalDebit - creditAmount;
            if (difference > this.tolerance) return null;

            return {
                debit_indices: match.map(m => m.orig_index),
                debit_dates: match.map(m => m.Date),
                debit_amounts: match.map(m => m.Debit),
                credit_indices: [creditRow.orig_index],
                credit_dates: [creditDate],
                credit_amounts: [creditAmount],
                total_debit: totalDebit,
                difference: difference,
                match_type: `DEBIT Combination ${match.length}`
            };
        }
        return null;
    }

    findCombinationRecursive(target, candidates, maxCombinations, tolerance) {
        const stack = [[0, 0, []]];
        while (stack.length > 0) {
            const [idx, currentSum, path] = stack.pop();

            if (Math.abs(target - currentSum) <= tolerance && path.length > 1) {
                return path;
            }
            if (path.length >= maxCombinations || idx >= candidates.length) continue;

            stack.push([idx + 1, currentSum, path]);

            const cand = candidates[idx];
            const newSum = currentSum + cand.Debit;
            if (newSum <= target + tolerance) {
                const newPath = [...path, cand];
                if (Math.abs(target - newSum) <= tolerance && newPath.length > 1) {
                    return newPath;
                }
                stack.push([idx + 1, newSum, newPath]);
            }
        }
        return null;
    }

    reconcileResidualRecovery() {
        const forcedMatches = this.matches.filter(m => m.is_forced);
        if (forcedMatches.length === 0) return;

        const unusedCredits = this.credit_df.filter(c => !this.used_credit_indices.has(c.orig_index));
        for (const match of forcedMatches) {
            const diff = match.difference || 0;
            if (diff === 0) continue;

            for (const creditRow of unusedCredits) {
                if (creditRow.Credit >= diff - this.residual_threshold) {
                    this.registerMatch({
                        debit_indices: [creditRow.orig_index], debit_dates: [creditRow.Date], debit_amounts: [creditRow.Credit],
                        credit_indices: [], credit_dates: [], credit_amounts: [],
                        total_debit: creditRow.Credit, difference: creditRow.Credit - diff,
                        match_type: `Residual Recovery (+${(diff / 100).toFixed(2)})`,
                        pass_name: "Residual Recovery"
                    });
                    break;
                }
            }
        }
    }

    run(rows) {
        this.separateMovements(rows);

        if (this.algorithm === "auto") {
            this.algorithm = "progressive_balance";
        }

        if (this.algorithm === "progressive_balance") {
            if (this.progressCallback) this.progressCallback(40, "Esecuzione Progressive Balance...");
            this.reconcileProgressiveBalance();
        } else if (this.algorithm === "subset_sum") {
            this.reconcileSubsetSum();
        } else if (this.algorithm === "greedy_amount_first") {
            this.reconcileGreedyAmountFirst();
        }

        if (this.progressCallback) this.progressCallback(75, "Recupero residui da blocchi forzati...");
        this.reconcileResidualRecovery();

        this.debit_df.forEach(r => r.used = this.used_debit_indices.has(r.orig_index));
        this.credit_df.forEach(r => r.used = this.used_credit_indices.has(r.orig_index));

        return this.getStats();
    }

    getStats() {
        const numDebitTot = this.debit_df.length;
        const amtDebitTot = this.debit_df.reduce((s, r) => s + r.Debit, 0);
        const numDebitUsed = Array.from(this.used_debit_indices).length;
        const amtDebitUsed = this.debit_df.filter(r => r.used).reduce((s, r) => s + r.Debit, 0);

        const numCreditTot = this.credit_df.length;
        const amtCreditTot = this.credit_df.reduce((s, r) => s + r.Credit, 0);
        const numCreditUsed = Array.from(this.used_credit_indices).length;
        const amtCreditUsed = this.credit_df.filter(r => r.used).reduce((s, r) => s + r.Credit, 0);

        const unusedDebitAmt = (amtDebitTot - amtDebitUsed) / 100;
        const unreconciledCreditAmt = (amtCreditTot - amtCreditUsed) / 100;

        return {
            "Total Receipts (DEBIT)": numDebitTot,
            "Used Receipts (DEBIT)": numDebitUsed,
            "% Used Receipts (DEBIT) (Num)": `${numDebitTot > 0 ? (numDebitUsed / numDebitTot * 100).toFixed(1) : 0}%`,
            "% Covered Receipts (DEBIT) (Vol)": `${amtDebitTot > 0 ? (amtDebitUsed / amtDebitTot * 100).toFixed(1) : 0}%`,
            "Unused Receipts (DEBIT)": numDebitTot - numDebitUsed,
            "Total Deposits (CREDIT)": numCreditTot,
            "Reconciled Deposits (CREDIT)": numCreditUsed,
            "% Reconciled Deposits (CREDIT) (Num)": `${numCreditTot > 0 ? (numCreditUsed / numCreditTot * 100).toFixed(1) : 0}%`,
            "% Covered Deposits (CREDIT) (Vol)": `${amtCreditTot > 0 ? (amtCreditUsed / amtCreditTot * 100).toFixed(1) : 0}%`,
            "Unreconciled Deposits (CREDIT)": numCreditTot - numCreditUsed,
            "Final delta (DEBIT - CREDIT)": `${(unusedDebitAmt - unreconciledCreditAmt).toFixed(2)} €`,
            "Structural Imbalance (Source)": `${((amtDebitTot - amtCreditTot) / 100).toFixed(2)} €`,
            _raw_debit_amount_perc: amtDebitTot > 0 ? (amtDebitUsed / amtDebitTot * 100) : 0,
            _raw_credit_amount_perc: amtCreditTot > 0 ? (amtCreditUsed / amtCreditTot * 100) : 0
        };
    }
}

class JSExcelReporter {
    constructor(engine, originalRows) {
        this.engine = engine;
        this.originalRows = originalRows;
    }

    async generateReport(progressCallback = null) {
        if (progressCallback) progressCallback(82, "Generazione fogli Excel (Summary, Matches, Original)...");
        const workbook = new ExcelJS.Workbook();

        this.buildMatchGroups();
        this.remapMatchesToSheetRows();
        this.buildMatchGroups();

        this.createSummarySheet(workbook);
        this.createManualSheet(workbook);
        this.createMatchesSheet(workbook);
        this.createAnomaliesSheet(workbook);
        this.createUnreconciledSheets(workbook);
        this.createOriginalSheet(workbook);
        this.createMonthlyBalanceSheet(workbook);

        if (progressCallback) progressCallback(95, "Finalizzazione e compressione file Excel...");
        const buffer = await workbook.xlsx.writeBuffer();
        return new Blob([buffer], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
    }

    buildMatchGroups() {
        this.groups = {};
        const GROUP_FILLS = ["DDEBF7", "C6EFCE", "FFD966"];

        this.engine.matches.forEach((row, groupIdx) => {
            const colorIdx = groupIdx % GROUP_FILLS.length;
            const transactionId = row["Transaction ID"] || `G${groupIdx + 1}`;
            const diff = row.difference || 0;

            (row.debit_indices || []).forEach((oi, idx) => {
                if (!this.groups[oi]) this.groups[oi] = { memberships: [] };
                this.groups[oi].memberships.push({
                    group_id: groupIdx + 1, color_idx: colorIdx, transaction_id: transactionId, difference: diff, side: 'debit', amount: (row.debit_amounts || [])[idx] || 0
                });
            });

            (row.credit_indices || []).forEach((oi, idx) => {
                if (!this.groups[oi]) this.groups[oi] = { memberships: [] };
                this.groups[oi].memberships.push({
                    group_id: groupIdx + 1, color_idx: colorIdx, transaction_id: transactionId, difference: diff, side: 'credit', amount: (row.credit_amounts || [])[idx] || 0
                });
            });
        });

        for (const oi in this.groups) {
            const primary = this.groups[oi].memberships[0];
            this.groups[oi].group_id = primary.group_id;
            this.groups[oi].color_idx = primary.color_idx;
            this.groups[oi].transaction_id = primary.transaction_id;
            this.groups[oi].difference = primary.difference;
            this.groups[oi].side = primary.side;
        }
    }

    buildOriginalLayout() {
        const expanded = [];
        const meta = [];
        const portionRows = {};
        let excelRow = 1;

        let monthKey = null;
        let monthDeb = 0, monthCre = 0;
        let monthDates = [];

        const monthlyEco = this.engine.computeMonthlyTotals();
        const monthNamesIt = ["Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno", "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"];

        const flushMonthTotal = () => {
            if (monthDates.length > 0) {
                const end = new Date(Math.max(...monthDates.map(d => d.getTime())));
                const totalDate = new Date(Date.UTC(end.getFullYear(), end.getMonth() + 1, 0));
                const label = `TOTALE MESE ${monthNamesIt[end.getMonth()]} ${end.getFullYear()}`;

                const m = monthKey !== null ? monthlyEco[monthKey] : undefined;
                const totDeb = m ? m.Debit : monthDeb;
                const totCre = m ? m.Credit : monthCre;

                expanded.push({ Date: totalDate, Debit: totDeb, Credit: totCre });
                excelRow++;
                meta.push({
                    month_total: true, group_label: label, color_idx: null, side: null, difference: totDeb - totCre, inserted: false
                });
            }
            monthDeb = 0; monthCre = 0; monthDates = [];
        };

        for (const r of this.originalRows) {
            const oi = r.orig_index;
            const date = r.Date;
            if (date) {
                const curMonth = `${date.getFullYear()}-${date.getMonth()}`;
                if (monthKey !== null && curMonth !== monthKey) flushMonthTotal();
                monthKey = curMonth;
                monthDates.push(date);
                monthDeb += (r.Debit || 0);
                monthCre += (r.Credit || 0);
            }

            const info = this.groups[oi];
            if (info && info.side === "debit") {
                const members = info.memberships;
                const portions = members.map(m => [m.amount, m]);
                const sumPortions = members.reduce((s, m) => s + m.amount, 0);
                const leftover = r.Debit - sumPortions;
                if (leftover > 0) portions.push([leftover, null]);

                const rowsThis = [];
                portions.forEach(([amount, member], i) => {
                    const rowCopy = { ...r, Debit: amount };
                    const inserted = i > 0;
                    if (inserted) rowCopy["Saldo Prog."] = null;
                    expanded.push(rowCopy);
                    excelRow++;
                    meta.push({
                        color_idx: member ? member.color_idx : null,
                        side: member ? member.side : null,
                        difference: member ? member.difference : null,
                        group_label: member ? member.transaction_id : "",
                        inserted: inserted
                    });
                    rowsThis.push(excelRow);
                });
                portionRows[oi] = rowsThis;
            } else {
                expanded.push({ ...r });
                excelRow++;
                meta.push({
                    color_idx: info ? info.color_idx : null,
                    side: info ? info.side : null,
                    difference: info ? info.difference : null,
                    group_label: info ? info.transaction_id : "",
                    inserted: false
                });
                portionRows[oi] = [excelRow];
            }
        }
        flushMonthTotal();

        return { expanded, meta, portionRows };
    }

    remapMatchesToSheetRows() {
        const { portionRows } = this.buildOriginalLayout();
        const gidToPos = {};
        for (const oi in this.groups) {
            this.groups[oi].memberships.forEach((m, pos) => {
                if (!gidToPos[oi]) gidToPos[oi] = {};
                gidToPos[oi][m.group_id] = pos;
            });
        }
        this.engine.matches.forEach((row, groupIdx) => {
            const gid = groupIdx + 1;
            const dRows = (row.debit_indices || []).map(oi => {
                const rows = portionRows[oi] || [oi + 2];
                const pos = (gidToPos[oi] || {})[gid];
                return rows[(pos !== undefined && pos < rows.length) ? pos : 0];
            });
            const cRows = (row.credit_indices || []).map(oi => (portionRows[oi] || [oi + 2])[0]);
            row["Transaction ID"] = `D(${dRows.join(",")})_A(${cRows.join(",")})`;
            row.debit_display = dRows;
            row.credit_display = cRows;
        });
    }

    createSummarySheet(workbook) {
        const ws = workbook.addWorksheet("Summary");
        ws.getCell("A1").value = "Reconciliation Summary";
        ws.getCell("A1").font = { bold: true, size: 16, color: { argb: "1F4E78" } };

        const stats = this.engine.getStats();
        ws.getCell("A3").value = "Key Performance Indicators";
        ws.getCell("A3").font = { bold: true, size: 12, color: { argb: "1F4E78" } };

        const fmtEur = (v) => v.toLocaleString("it-IT", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + " €";

        const kpis = [
            ["Debit Coverage (Volume)", `${stats._raw_debit_amount_perc.toFixed(2)}%`],
            ["Credit Coverage (Volume)", `${stats._raw_credit_amount_perc.toFixed(2)}%`],
            ["Unreconciled Debits", stats["Unused Receipts (DEBIT)"]],
            ["Unreconciled Credits", stats["Unreconciled Deposits (CREDIT)"]],
            ["Final Delta", stats["Final delta (DEBIT - CREDIT)"]]
        ];

        const saldo = this.analyzeSaldoProg();
        if (saldo.present) {
            kpis.push(["Saldo Prog. Iniziale (Cassa)", fmtEur(saldo.opening)]);
            kpis.push(["Saldo Prog. Finale (Cassa)", fmtEur(saldo.closing)]);
            kpis.push(["Variazione Cassa (Dare - Avere)", fmtEur(saldo.closing - saldo.opening)]);
            if (saldo.negative_rows > 0) kpis.push(["⚠ Righe con Saldo Negativo", saldo.negative_rows]);
            if (saldo.inconsistent_rows > 0) kpis.push(["⚠ Righe Saldo Incoerenti", saldo.inconsistent_rows]);
        }

        kpis.forEach(([label, val], idx) => {
            const r = idx + 4;
            ws.getCell(`A${r}`).value = label;
            ws.getCell(`A${r}`).font = { bold: true };
            ws.getCell(`B${r}`).value = val;
        });

        const cursor = 4 + kpis.length + 1;
        ws.getCell(`A${cursor}`).value = "Automated Analysis";
        ws.getCell(`A${cursor}`).font = { bold: true, size: 12, color: { argb: "1F4E78" } };

        let summaryText = `Reconciliation resulted in ${stats._raw_debit_amount_perc.toFixed(1)}% of debit volume and ${stats._raw_credit_amount_perc.toFixed(1)}% of credit volume being matched. `;
        if (saldo.present) {
            summaryText += `Cash balance (Saldo Prog.): ${fmtEur(saldo.opening)} at start, ${fmtEur(saldo.closing)} at end. `;
            if (saldo.inconsistent_rows > 0) {
                summaryText += `WARNING: ${saldo.inconsistent_rows} rows show an inconsistent progressive balance. `;
            } else if (saldo.negative_rows > 0) {
                summaryText += `WARNING: ${saldo.negative_rows} rows show a negative cash balance. `;
            }
        }
        if (stats._raw_debit_amount_perc > 95 && stats._raw_credit_amount_perc > 95) {
            summaryText += "This is a great result, with very few items left to check.";
        }
        ws.getCell(`A${cursor + 1}`).value = summaryText;

        ws.getColumn("A").width = 30;
        ws.getColumn("B").width = 20;
    }

    analyzeSaldoProg() {
        const rows = this.originalRows || [];
        const result = { present: false };
        if (rows.length === 0 || !("Saldo Prog." in rows[0])) return result;

        const saldos = rows.map(r => {
            const v = r["Saldo Prog."];
            return (v === undefined || v === null || v === "") ? NaN : Number(v);
        });
        if (saldos.every(s => isNaN(s)) || isNaN(saldos[0])) return result;

        const saldoCents = saldos.map(s => Math.round(s * 100));
        const first = rows[0];
        const opening = saldoCents[0] - (first.Debit - first.Credit);
        const checkMap = {};
        let acc = opening;
        let inconsistentRows = 0;
        let negativeRows = 0;
        rows.forEach((r, i) => {
            acc += (r.Debit - r.Credit);
            const mismatch = Math.abs(acc - saldoCents[i]);
            if (saldos[i] < 0) negativeRows++;
            if (mismatch > 1) {
                inconsistentRows++;
                checkMap[r.orig_index] = `⚠️ Saldo incoerente (Δ ${(mismatch / 100).toLocaleString("it-IT", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} €)`;
            } else {
                checkMap[r.orig_index] = "";
            }
        });

        result.present = true;
        result.opening = opening / 100;
        result.closing = saldoCents[saldoCents.length - 1] / 100;
        result.min = Math.min(...saldos.filter(s => !isNaN(s)));
        result.negative_rows = negativeRows;
        result.inconsistent_rows = inconsistentRows;
        result.check_map = checkMap;
        return result;
    }

    createManualSheet(workbook) {
        const ws = workbook.addWorksheet("MANUAL");
        ws.getCell("A1").value = "CashRec Manual & Parameters";
        ws.getCell("A1").font = { bold: true, size: 14 };

        const params = [
            ["Tolerance", `${(this.engine.tolerance / 100).toFixed(2)} €`],
            ["Time Window", `${this.engine.days_window} days`],
            ["Max Combinations", `${this.engine.max_combinations}`],
            ["Search Direction", this.engine.search_direction]
        ];

        params.forEach(([k, v], i) => {
            ws.getCell(`A${i + 3}`).value = k;
            ws.getCell(`B${i + 3}`).value = v;
        });

        ws.getColumn("A").width = 25;
        ws.getColumn("B").width = 20;
    }

    createMatchesSheet(workbook) {
        const ws = workbook.addWorksheet("Matches");
        const headers = ["Transaction ID", "debit_indices", "debit_dates", "debit_amounts", "total_debit", "credit_date", "num_credits", "credit_indices", "credit_amounts", "total_credit", "difference", "match_type", "pass_name"];
        ws.addRow(headers).font = { bold: true };

        this.engine.matches.forEach(m => {
            ws.addRow([
                m["Transaction ID"],
                (m.debit_display || m.debit_indices).join(", "),
                (m.debit_dates || []).map(d => d.toLocaleDateString("it-IT")).join(", "),
                (m.debit_amounts || []).map(a => (a / 100).toFixed(2)).join(", "),
                m.total_debit / 100,
                m.credit_date ? m.credit_date.toLocaleDateString("it-IT") : "",
                m.num_credits,
                (m.credit_display || m.credit_indices).join(", "),
                (m.credit_amounts || []).map(a => (a / 100).toFixed(2)).join(", "),
                m.total_credit / 100,
                m.difference / 100,
                m.match_type,
                m.pass_name
            ]);
        });
    }

    createAnomaliesSheet(workbook) {
        const anomalies = this.engine.matches.filter(m => (m.match_type || "").startsWith("ANOMALY"));
        if (anomalies.length === 0) return;

        const ws = workbook.addWorksheet("Anomalie");
        const headers = ["Transaction ID", "credit_date", "total_credit", "debit_indices", "debit_dates", "debit_amounts", "total_debit", "uncovered", "match_type"];
        ws.addRow(headers).font = { bold: true };

        anomalies.forEach(m => {
            ws.addRow([
                m["Transaction ID"],
                m.credit_date ? m.credit_date.toLocaleDateString("it-IT") : "",
                m.total_credit / 100,
                (m.debit_display || m.debit_indices).join(", "),
                (m.debit_dates || []).map(d => d.toLocaleDateString("it-IT")).join(", "),
                (m.debit_amounts || []).map(a => (a / 100).toFixed(2)).join(", "),
                m.total_debit / 100,
                m.difference / 100,
                m.match_type
            ]);
        });
    }

    createUnreconciledSheets(workbook) {
        const unusedDebits = this.engine.debit_df.filter(r => !r.used);
        if (unusedDebits.length > 0) {
            const ws = workbook.addWorksheet("Unused DEBIT");
            ws.addRow(["Row Index", "Date", "Amount"]).font = { bold: true };
            unusedDebits.forEach(r => {
                ws.addRow([r.orig_index + 2, r.Date.toLocaleDateString("it-IT"), r.Debit / 100]);
            });
        }

        const unreconciledCredits = this.engine.credit_df.filter(r => !r.used);
        if (unreconciledCredits.length > 0) {
            const ws = workbook.addWorksheet("Unreconciled CREDIT");
            ws.addRow(["Row Index", "Date", "Amount"]).font = { bold: true };
            unreconciledCredits.forEach(r => {
                ws.addRow([r.orig_index + 2, r.Date.toLocaleDateString("it-IT"), r.Credit / 100]);
            });
        }
    }

    createOriginalSheet(workbook) {
        const ws = workbook.addWorksheet("Original");
        const { expanded, meta } = this.buildOriginalLayout();

        const headers = ["Data", "Data Valuta", "Dare", "Avere", "Gruppo", "Delta"];
        if (this.originalRows.some(r => r["Saldo Prog."] !== undefined)) {
            headers.splice(4, 0, "Saldo Prog.");
        }
        ws.addRow(headers).font = { bold: true };

        const GROUP_FILLS = ["DDEBF7", "C6EFCE", "FFD966"];

        expanded.forEach((r, idx) => {
            const m = meta[idx];
            const rowData = [
                r.Date ? r.Date.toLocaleDateString("it-IT") : "",
                r.valuta_date ? r.valuta_date.toLocaleDateString("it-IT") : "",
                r.Debit ? r.Debit / 100 : 0,
                r.Credit ? r.Credit / 100 : 0
            ];

            if (headers.includes("Saldo Prog.")) {
                rowData.push(r["Saldo Prog."] !== undefined && r["Saldo Prog."] !== null ? r["Saldo Prog."] : "");
            }

            rowData.push(m.group_label);
            rowData.push(m.difference !== null && m.difference !== undefined ? m.difference / 100 : "");

            const rowCell = ws.addRow(rowData);

            [3, 4, headers.length].forEach(ci => {
                if (rowCell.getCell(ci).value !== "") rowCell.getCell(ci).numFmt = '#,##0.00 €';
            });

            if (m.month_total) {
                rowCell.eachCell(cell => {
                    cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'E7E6E6' } };
                    cell.font = { bold: true };
                });
            } else if (m.color_idx !== null && m.color_idx !== undefined) {
                const fillHex = GROUP_FILLS[m.color_idx];
                if (m.side === "debit") {
                    rowCell.getCell(3).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: fillHex } };
                } else if (m.side === "credit") {
                    rowCell.getCell(4).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: fillHex } };
                }
                const diffCell = rowCell.getCell(headers.length);
                diffCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: fillHex } };
                if (m.difference > 0) diffCell.font = { bold: true, color: { argb: "C00000" } };
            }
        });

        this.createOriginalLegend(ws, expanded.length);
    }

    createOriginalLegend(ws, nRows) {
        const groupNames = ["Blu", "Verde", "Arancione"];
        const groupFills = ["DDEBF7", "C6EFCE", "FFD966"];
        let row = nRows + 3;

        ws.getCell(`A${row}`).value = "Legenda: le celle Dare/Avere con lo stesso colore appartengono allo stesso gruppo di abbinamento (vedi foglio 'Matches'). Il colore si ripete ogni 3 gruppi.";
        ws.getCell(`A${row}`).font = { bold: true, color: { argb: "1F4E78" } };
        row += 1;

        groupFills.forEach((fill, idx) => {
            const cell = ws.getCell(`${String.fromCharCode(65 + idx)}${row}`);
            cell.value = `Gruppo ${idx + 1} (mod 3) - ${groupNames[idx]}`;
            cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: fill } };
        });
        row += 2;

        ws.getCell(`A${row}`).value = "Colonna 'Delta': Δ in € del gruppo (0 = gruppo quadrato; > 0 = differenza/residuo da verificare). Le righe 'TOTALE MESE' usano una finestra economica 'lasca': i versamenti dei primi giorni del mese successivo relativi al mese precedente (data valuta / aggancio) vengono riportati al mese di competenza, per quadrare il mese pochi giorni dopo l'inizio del successivo.";
        row += 1;
        ws.getCell(`A${row}`).value = "Righe inserite: se un incasso (Dare) è ripartito su più versamenti, la riga originale mostra la quota consumata dal PRIMO versamento (in ordine del foglio 'Matches'); sotto di essa viene inserita una nuova riga (stessa data) per ogni quota residua, ciascuna con il colore del proprio gruppo.";
    }

    createMonthlyBalanceSheet(workbook) {
        const ws = workbook.addWorksheet("Quadratura Mensile");
        const headers = ["Mese", "Dare (Incassi)", "Avere (Versamenti)", "Δ Mese", "Cumulato", "Stato"];
        const headerRow = ws.addRow(headers);
        headerRow.font = { bold: true };
        headerRow.eachCell(cell => {
            cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF4472C4' } };
            cell.font = { bold: true, color: { argb: 'FFFFFFFF' } };
        });

        const monthlyEco = this.engine.computeMonthlyTotals();
        const tolerance = this.engine.tolerance / 100;
        const rows = [];
        let cumulato = 0;
        Object.keys(monthlyEco).sort().forEach(k => {
            const m = monthlyEco[k];
            const [y, mm] = k.split("-").map(Number);
            const dare = m.Debit / 100;
            const avere = m.Credit / 100;
            const delta = dare - avere;
            cumulato += delta;
            const stato = Math.abs(delta) <= tolerance ? "OK" : "Controllare";
            rows.push([`${y}-${String(mm + 1).padStart(2, '0')}`, dare, avere, delta, cumulato, stato]);
        });

        rows.forEach(r => {
            const rowCell = ws.addRow(r);
            const dareCell = rowCell.getCell(2);
            const avereCell = rowCell.getCell(3);
            const deltaCell = rowCell.getCell(4);
            const cumCell = rowCell.getCell(5);
            const statoCell = rowCell.getCell(6);
            [dareCell, avereCell, deltaCell, cumCell].forEach(c => c.numFmt = '#,##0.00 €');
            if (r[3] !== undefined && Math.abs(r[3]) <= tolerance) {
                statoCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFC6EFCE' } };
            } else {
                statoCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFFC7CE' } };
            }
            cumCell.fill = {
                type: 'pattern', pattern: 'solid',
                fgColor: { argb: cumulato >= 0 ? 'FFC6EFCE' : 'FFFFC7CE' }
            };
        });

        ws.getColumn(1).width = 12;
        for (let c = 2; c <= 6; c++) ws.getColumn(c).width = 20;
    }
}

self.onmessage = async function(e) {
    const { type, data } = e.data;
    if (type === 'start') {
        const { fileBuffer, fileName, columnMappingForm, engineParams } = data;
        const logMessages = ["=== CashRec Log Elaborazione ===", "Data: " + new Date().toLocaleString('it-IT'), "File: " + fileName + "\n"];

        try {
            self.postMessage({ type: 'progress', percent: 5, statusText: 'Caricamento librerie e lettura file Excel...' });

            const dummyFile = { arrayBuffer: async () => fileBuffer };

            const rows = await readExcelFile(dummyFile, fileName, columnMappingForm, logMessages, (perc, msg) => {
                self.postMessage({ type: 'progress', percent: perc, statusText: msg });
            });

            self.postMessage({ type: 'progress', percent: 30, statusText: "Esecuzione motore di riconciliazione..." });
            engineParams.progressCallback = (perc, msg) => {
                self.postMessage({ type: 'progress', percent: perc, statusText: msg });
            };

            logMessages.push("[Algoritmo] Parametri: " + JSON.stringify(engineParams, null, 2));
            const engine = new JSReconciliationEngine(engineParams);
            const stats = engine.run(rows);
            logMessages.push("[Algoritmo] Elaborazione completata.");
            logMessages.push("[Statistiche]\n" + JSON.stringify(stats, null, 2));

            self.postMessage({ type: 'progress', percent: 80, statusText: "Generazione del report Excel..." });
            const reporter = new JSExcelReporter(engine, rows);
            const reportBlob = await reporter.generateReport((perc, msg) => {
                self.postMessage({ type: 'progress', percent: perc, statusText: msg });
            });

            self.postMessage({ type: 'progress', percent: 100, statusText: "Elaborazione completata!" });

            const origName = fileName.replace(/\.[^/.]+$/, "");
            const reportFilename = origName + "_result.xlsx";
            logMessages.push("[Report] Report Excel generato con successo (" + reportFilename + ").");

            const fullLogText = logMessages.join('\n');

            const monthlyTotalsRaw = engine.computeMonthlyTotals();
            const monthNamesIt = ["Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno", "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"];
            const monthlyBalance = Object.keys(monthlyTotalsRaw).sort().map(k => {
                const [y, m] = k.split("-").map(Number);
                const d = monthlyTotalsRaw[k].Debit / 100;
                const c = monthlyTotalsRaw[k].Credit / 100;
                return {
                    key: k,
                    label: (monthNamesIt[m] || ('Mese ' + (m + 1))) + ' ' + y,
                    debit: d,
                    credit: c,
                    delta: d - c
                };
            });

            let countExact = 0;
            let countTolerance = 0;
            let countAnomaly = 0;
            let countForced = 0;

            engine.matches.forEach(m => {
                const isAnomaly = (m.match_type || '').startsWith('ANOMALY');
                if (isAnomaly) {
                    countAnomaly++;
                } else if (m.difference === 0) {
                    countExact++;
                } else if (m.difference <= engine.tolerance) {
                    countTolerance++;
                } else if (m.is_forced) {
                    countForced++;
                } else {
                    countExact++;
                }
            });

            const unusedDebits = engine.debit_df.filter(r => !r.used);
            const unreconciledCredits = engine.credit_df.filter(r => !r.used);

            const matchCounts = {
                exact: countExact,
                tolerance: countTolerance,
                anomaly: countAnomaly,
                forced: countForced,
                unused_debit: unusedDebits.length,
                unreconciled_credit: unreconciledCredits.length
            };

            const tableItems = [];

            engine.matches.filter(m => (m.match_type || '').startsWith('ANOMALY')).forEach(m => {
                const dStr = m.credit_date ? m.credit_date.toLocaleDateString('it-IT') : '';
                const dateVal = m.credit_date ? m.credit_date.getTime() : 0;
                tableItems.push({
                    type: 'anomaly',
                    typeLabel: 'Anomalia',
                    badgeClass: 'bg-danger-subtle text-danger border border-danger-subtle',
                    dateStr: dStr,
                    dateVal: dateVal,
                    transactionId: m["Transaction ID"],
                    amount: m.total_credit / 100,
                    diff: m.difference / 100,
                    details: m.match_type
                });
            });

            unusedDebits.forEach(r => {
                const dStr = r.Date ? r.Date.toLocaleDateString('it-IT') : '';
                const dateVal = r.Date ? r.Date.getTime() : 0;
                tableItems.push({
                    type: 'unused_debit',
                    typeLabel: 'Incasso non usato',
                    badgeClass: 'bg-warning-subtle text-warning border border-warning-subtle',
                    dateStr: dStr,
                    dateVal: dateVal,
                    transactionId: 'Riga ' + (r.orig_index + 2),
                    amount: r.Debit / 100,
                    diff: 0,
                    details: 'Incasso in Dare senza versamento abbinato'
                });
            });

            unreconciledCredits.forEach(r => {
                const dStr = r.Date ? r.Date.toLocaleDateString('it-IT') : '';
                const dateVal = r.Date ? r.Date.getTime() : 0;
                tableItems.push({
                    type: 'unreconciled_credit',
                    typeLabel: 'Versamento scoperto',
                    badgeClass: 'bg-info-subtle text-info border border-info-subtle',
                    dateStr: dStr,
                    dateVal: dateVal,
                    transactionId: 'Riga ' + (r.orig_index + 2),
                    amount: r.Credit / 100,
                    diff: r.Credit / 100,
                    details: 'Versamento in Avere senza incassi a copertura'
                });
            });

            self.postMessage({
                type: 'success',
                reportBlob: reportBlob,
                reportFilename: reportFilename,
                fullLogText: fullLogText,
                stats: stats,
                dashboardData: {
                    monthlyBalance: monthlyBalance,
                    matchCounts: matchCounts,
                    tableItems: tableItems
                }
            });
        } catch (err) {
            self.postMessage({ type: 'error', phase: 'Elaborazione Client-Side', message: err.message || String(err) });
        }
    }
};
