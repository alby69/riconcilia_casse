export class JSReconciliationEngine {
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
