COPY (
WITH chronic_diagnoses AS (
    SELECT
        d.hadm_id,
        COUNT(*) AS chronic_disease_count
    FROM mimiciv_hosp.diagnoses_icd d
    WHERE
        (d.icd_version = 9 AND d.icd_code IN (SELECT icd_code FROM cci_icd9_chronic))
        OR
        (d.icd_version = 10 AND d.icd_code IN (SELECT icd_code FROM cci_icd10_chronic))
    GROUP BY d.hadm_id
)
SELECT
    a.hadm_id,
    a.subject_id,
    COALESCE(cd.chronic_disease_count, 0) AS chronic_disease_count
FROM mimiciv_hosp.admissions a
LEFT JOIN chronic_diagnoses cd ON a.hadm_id = cd.hadm_id
)
TO 'C:/Users/Lenovo/Desktop/Msc Project/data/features/chronic_disease_count.csv'
WITH CSV HEADER;