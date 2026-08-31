COPY (
WITH first_icu AS (
    SELECT hadm_id, outtime AS prediction_point
    FROM (
        SELECT
            hadm_id,
            outtime,
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
        -- removed outtime IS NOT NULL here so we capture all first stays
    ) t
    WHERE rn = 1
),
transfers_prefix AS (
    SELECT
        t.hadm_id,
        t.careunit,
        t.intime,
        LEAST(t.outtime, fi.prediction_point) AS capped_outtime
    FROM mimiciv_hosp.transfers t
    INNER JOIN first_icu fi ON t.hadm_id = fi.hadm_id
    -- only process if prediction_point is not null and transfer started before it
    WHERE fi.prediction_point IS NOT NULL
    AND t.intime < fi.prediction_point
    AND t.outtime IS NOT NULL
),
process_features AS (
    SELECT
        hadm_id,
        SUM(GREATEST(0, EXTRACT(EPOCH FROM (capped_outtime - intime)) / 3600))
            AS accumulated_duration_hrs,
        SUM(CASE WHEN careunit IN (
            'Medical Intensive Care Unit (MICU)',
            'Cardiac Vascular Intensive Care Unit (CVICU)',
            'Medical/Surgical Intensive Care Unit (MICU/SICU)',
            'Surgical Intensive Care Unit (SICU)',
            'Trauma SICU (TSICU)',
            'Coronary Care Unit (CCU)',
            'Neuro Surgical Intensive Care Unit (Neuro SICU)',
            'Intensive Care Unit (ICU)'
        ) THEN GREATEST(0, EXTRACT(EPOCH FROM (capped_outtime - intime)) / 3600)
        ELSE 0 END) AS icu_duration_hrs,
        SUM(CASE WHEN careunit IN (
            'Surgery', 'Surgery/Trauma', 'Cardiac Surgery',
            'Thoracic Surgery', 'Surgery/Pancreatic/Biliary/Bariatric', 'Med/Surg/Trauma'
        ) THEN GREATEST(0, EXTRACT(EPOCH FROM (capped_outtime - intime)) / 3600)
        ELSE 0 END) AS sw_duration_hrs,
        COUNT(CASE WHEN careunit IN (
            'Medical Intensive Care Unit (MICU)',
            'Cardiac Vascular Intensive Care Unit (CVICU)',
            'Medical/Surgical Intensive Care Unit (MICU/SICU)',
            'Surgical Intensive Care Unit (SICU)',
            'Trauma SICU (TSICU)',
            'Coronary Care Unit (CCU)',
            'Neuro Surgical Intensive Care Unit (Neuro SICU)',
            'Intensive Care Unit (ICU)'
        ) THEN 1 END) AS num_icu_stays,
        COUNT(CASE WHEN careunit IN (
            'Surgery', 'Surgery/Trauma', 'Cardiac Surgery',
            'Thoracic Surgery', 'Surgery/Pancreatic/Biliary/Bariatric', 'Med/Surg/Trauma'
        ) THEN 1 END) AS num_sw_stays,
        COUNT(*) AS total_events
    FROM transfers_prefix
    GROUP BY hadm_id
)
SELECT
    a.hadm_id,
    a.subject_id,
    COALESCE(a.admission_location, 'INFORMATION NOT AVAILABLE') AS admission_location,
    COALESCE(pf.accumulated_duration_hrs, 0) AS accumulated_duration_hrs,
    COALESCE(pf.icu_duration_hrs, 0)         AS icu_duration_hrs,
    COALESCE(pf.sw_duration_hrs, 0)          AS sw_duration_hrs,
    COALESCE(pf.num_icu_stays, 0)            AS num_icu_stays,
    COALESCE(pf.num_sw_stays, 0)             AS num_sw_stays,
    COALESCE(pf.total_events, 0)             AS total_events
FROM mimiciv_hosp.admissions a
INNER JOIN first_icu fi ON a.hadm_id = fi.hadm_id
LEFT JOIN process_features pf ON a.hadm_id = pf.hadm_id
)
TO 'C:/Users/Lenovo/Desktop/Msc Project/data/features/icu_process_features_v2.csv'
WITH CSV HEADER;