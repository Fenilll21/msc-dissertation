COPY (
WITH abnormal_labs AS (
    SELECT 
        hadm_id,
        COUNT(*) AS total_labs,
        SUM(CASE WHEN flag = 'abnormal' THEN 1 ELSE 0 END) AS abnormal_count,
        ROUND(100.0 * SUM(CASE WHEN flag = 'abnormal' THEN 1 ELSE 0 END) / 
              NULLIF(COUNT(*), 0), 2) AS pct_abnormal_labs
    FROM mimiciv_hosp.labevents
    WHERE hadm_id IS NOT NULL
    GROUP BY hadm_id
),
diagnosis_count AS (
    SELECT 
        hadm_id,
        COUNT(*) AS num_diagnoses
    FROM mimiciv_hosp.diagnoses_icd
    GROUP BY hadm_id
),
medication_count AS (
    SELECT 
        hadm_id,
        COUNT(DISTINCT drug) AS num_medications,
        CASE WHEN COUNT(DISTINCT drug) >= 5 THEN 1 ELSE 0 END AS polypharmacy
    FROM mimiciv_hosp.prescriptions
    WHERE hadm_id IS NOT NULL
    GROUP BY hadm_id
)
SELECT 
    a.hadm_id,
    COALESCE(al.total_labs, 0) AS total_labs,
    COALESCE(al.abnormal_count, 0) AS abnormal_labs,
    COALESCE(al.pct_abnormal_labs, 0) AS pct_abnormal_labs,
    COALESCE(dc.num_diagnoses, 0) AS num_diagnoses,
    COALESCE(mc.num_medications, 0) AS num_medications,
    COALESCE(mc.polypharmacy, 0) AS polypharmacy
FROM mimiciv_hosp.admissions a
INNER JOIN (
    SELECT DISTINCT hadm_id 
    FROM mimiciv_hosp.transfers
    WHERE careunit IN (
        'Surgery',
        'Surgery/Trauma',
        'Cardiac Surgery',
        'Thoracic Surgery',
        'Surgery/Pancreatic/Biliary/Bariatric',
        'Med/Surg/Trauma'
    )
) sw ON a.hadm_id = sw.hadm_id
LEFT JOIN abnormal_labs al ON a.hadm_id = al.hadm_id
LEFT JOIN diagnosis_count dc ON a.hadm_id = dc.hadm_id
LEFT JOIN medication_count mc ON a.hadm_id = mc.hadm_id
)
TO 'E:/sw_clinical_features.csv'
WITH CSV HEADER;
