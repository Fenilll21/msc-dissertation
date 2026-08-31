COPY (
SELECT
    a.hadm_id,
    a.subject_id,
    CASE
        WHEN p.anchor_age BETWEEN 18 AND 27 THEN '18-27'
        WHEN p.anchor_age BETWEEN 28 AND 37 THEN '28-37'
        WHEN p.anchor_age BETWEEN 38 AND 47 THEN '38-47'
        WHEN p.anchor_age BETWEEN 48 AND 57 THEN '48-57'
        WHEN p.anchor_age BETWEEN 58 AND 67 THEN '58-67'
        WHEN p.anchor_age BETWEEN 68 AND 77 THEN '68-77'
        WHEN p.anchor_age BETWEEN 78 AND 87 THEN '78-87'
        ELSE '88+'
    END AS age_band,
    p.gender,
    CASE
        WHEN a.insurance = 'Medicare' THEN 'Medicare'
        WHEN a.insurance = 'Medicaid' THEN 'Medicaid'
        WHEN a.insurance = 'Private'  THEN 'Private'
        ELSE 'Other'
    END AS insurance,
    CASE
        WHEN a.race ILIKE '%white%'
          OR a.race ILIKE '%portuguese%'        THEN 'WHITE'
        WHEN a.race ILIKE '%black%'              THEN 'BLACK'
        WHEN a.race ILIKE '%hispanic%'
          OR a.race ILIKE '%latino%'
          OR a.race ILIKE '%south american%'    THEN 'HISPANIC/LATINO'
        WHEN a.race ILIKE '%asian%'              THEN 'ASIAN'
        WHEN a.race ILIKE '%american indian%'
          OR a.race ILIKE '%alaska native%'
          OR a.race ILIKE '%native hawaiian%'
          OR a.race ILIKE '%pacific islander%'  THEN 'NATIVE'
        WHEN a.race ILIKE '%multiple%'           THEN 'MULTIPLE'
        WHEN a.race IS NULL
          OR a.race ILIKE '%unknown%'
          OR a.race ILIKE '%unable%'
          OR a.race ILIKE '%declined%'          THEN 'UNKNOWN'
        ELSE 'OTHER'
    END AS ethnicity,
    COALESCE(a.marital_status, 'UNKNOWN') AS marital_status

FROM mimiciv_hosp.admissions a
INNER JOIN mimiciv_hosp.patients p ON a.subject_id = p.subject_id
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
)
TO 'C:/Users/Lenovo/Desktop/Msc Project/data/features/sw_demographic_features_v2.csv'
WITH CSV HEADER;