export const DEFAULT_CONFIG = {
    common: {
        algorithm: "progressive_balance",
        tolerance: 50.0,
        days_window: 5,
        search_direction: "past_only",
        max_combinations: 10,
        residual_threshold: 50.0,
        residual_days_window: 5,
        handover_days: 5,
        column_mapping: {
            "Data Reg.": "Date",
            "Dare": "Debit",
            "Avere": "Credit"
        }
    }
};

export const DEFAULT_PROFILES = {
    "Operatore Punto Vendita (Default)": {
        algorithm: "progressive_balance",
        days_window: 5,
        tolerance: 50.0,
        search_direction: "past_only",
        max_combinations: 10,
        residual_threshold: 50.0,
        residual_days_window: 5,
        handover_days: 5
    },
    "Riconciliazione Mensile Fine Anno": {
        algorithm: "subset_sum",
        days_window: 15,
        tolerance: 1.0,
        search_direction: "both",
        max_combinations: 10,
        residual_threshold: 50.0,
        residual_days_window: 30
    },
    "Greedy Importi Elevati": {
        algorithm: "greedy_amount_first",
        days_window: 15,
        tolerance: 50.0,
        search_direction: "both",
        max_combinations: 10,
        residual_threshold: 50.0,
        residual_days_window: 5
    }
};

export const I18N_TRANSLATIONS = {
    it: {
        dropzone_title: "Trascina qui i tuoi file Excel o CSV",
        dropzone_sub: "supporta formati .xlsx, .xls, .csv (singolo o multi-file)",
        advanced_toggle: "Impostazioni Avanzate",
        btn_process: "Elabora File",
        btn_download_excel: "Scarica Report Excel",
        btn_download_pdf: "Scarica Report PDF",
        btn_new_file: "Nuovo File"
    },
    en: {
        dropzone_title: "Drag & drop your Excel or CSV files here",
        dropzone_sub: "supports .xlsx, .xls, .csv formats (single or multi-file)",
        advanced_toggle: "Advanced Settings",
        btn_process: "Process File",
        btn_download_excel: "Download Excel Report",
        btn_download_pdf: "Download PDF Report",
        btn_new_file: "New File"
    }
};
