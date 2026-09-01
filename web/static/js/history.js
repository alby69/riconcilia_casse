export class CashRecDB {
    static DB_NAME = 'CashRecDB';
    static DB_VERSION = 1;
    static STORE_NAME = 'history';

    static openDB() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(CashRecDB.DB_NAME, CashRecDB.DB_VERSION);
            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains(CashRecDB.STORE_NAME)) {
                    db.createObjectStore(CashRecDB.STORE_NAME, { keyPath: 'id', autoIncrement: true });
                }
            };
            request.onsuccess = (e) => resolve(e.target.result);
            request.onerror = (e) => reject(e.target.error);
        });
    }

    static async saveRun(runData) {
        try {
            const db = await CashRecDB.openDB();
            const tx = db.transaction(CashRecDB.STORE_NAME, 'readwrite');
            const store = tx.objectStore(CashRecDB.STORE_NAME);

            const all = await new Promise((res) => {
                const req = store.getAll();
                req.onsuccess = () => res(req.result || []);
            });

            if (all.length >= 10) {
                all.sort((a, b) => a.timestamp - b.timestamp);
                store.delete(all[0].id);
            }

            store.add({
                timestamp: Date.now(),
                dateStr: new Date().toLocaleString('it-IT'),
                fileName: runData.fileName,
                stats: runData.stats,
                dashboardData: runData.dashboardData,
                fullLogText: runData.fullLogText
            });
        } catch (err) {
            console.warn("IndexedDB save error:", err);
        }
    }

    static async getHistory() {
        try {
            const db = await CashRecDB.openDB();
            const tx = db.transaction(CashRecDB.STORE_NAME, 'readonly');
            const store = tx.objectStore(CashRecDB.STORE_NAME);
            return new Promise((resolve) => {
                const req = store.getAll();
                req.onsuccess = () => {
                    const res = req.result || [];
                    res.sort((a, b) => b.timestamp - a.timestamp);
                    resolve(res);
                };
            });
        } catch (err) {
            console.warn("IndexedDB get error:", err);
            return [];
        }
    }

    static async clearHistory() {
        try {
            const db = await CashRecDB.openDB();
            const tx = db.transaction(CashRecDB.STORE_NAME, 'readwrite');
            tx.objectStore(CashRecDB.STORE_NAME).clear();
        } catch (err) {
            console.warn("IndexedDB clear error:", err);
        }
    }
}
