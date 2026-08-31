COPY (
    SELECT hadm_id, outtime AS prediction_point
    FROM (
        SELECT hadm_id, outtime,
               ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) AS rn
        FROM mimiciv_hosp.transfers
        WHERE careunit IN (
            'Medical Intensive Care Unit (MICU)',
            'Cardiac Vascular Intensive Care Unit (CVICU)',
            'Medical/Surgical Intensive Care Unit (MICU/SICU)',
            'Surgical Intensive Care Unit (SICU)',
            'Trauma SICU (TSICU)',
            'Coronary Care Unit (CCU)',
            'Neuro Surgical Intensive Care Unit (Neuro SICU)',
            'Intensive Care Unit (ICU)'
        )
    ) t
    WHERE rn = 1
)
TO 'C:/Users/Lenovo/Desktop/Msc Project/data/features/icu_prediction_points.csv'
WITH CSV HEADER;