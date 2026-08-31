COPY (
    SELECT 
        hadm_id AS case_id,
        admittime AS event_time,
        'Hospital Admission' AS activity
    FROM mimiciv_hosp.admissions

    UNION ALL

    SELECT 
        hadm_id AS case_id,
        dischtime AS event_time,
        'Hospital Discharge' AS activity
    FROM mimiciv_hosp.admissions

    UNION ALL

    SELECT 
        hadm_id AS case_id,
        intime AS event_time,
        'ICU Admission' AS activity
    FROM mimiciv_icu.icustays

    UNION ALL

    SELECT 
        hadm_id AS case_id,
        outtime AS event_time,
        'ICU Discharge' AS activity
    FROM mimiciv_icu.icustays
)
TO 'E:/mimic_raw_events.csv'
WITH CSV HEADER;