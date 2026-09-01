export class JSExcelReporter {
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
