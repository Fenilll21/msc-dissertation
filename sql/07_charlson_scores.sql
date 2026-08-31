COPY (
WITH condition_flags AS (
    -- For each admission, flag which Charlson conditions are present
    -- Using LEFT on icd_code to handle prefix matching
    SELECT DISTINCT
        d.hadm_id,
        cm.condition,
        cm.weight
    FROM mimiciv_hosp.diagnoses_icd d
    INNER JOIN charlson_mapping cm
        ON d.icd_version = cm.icd_version
        AND LEFT(d.icd_code, LENGTH(cm.icd_code)) = cm.icd_code
),
charlson_scores AS (
    -- Sum weights per admission, counting each condition only once
    SELECT
        hadm_id,
        SUM(weight) AS charlson_score
    FROM condition_flags
    GROUP BY hadm_id
)
SELECT
    a.hadm_id,
    a.subject_id,
    COALESCE(cs.charlson_score, 0) AS charlson_score
FROM mimiciv_hosp.admissions a
LEFT JOIN charlson_scores cs ON a.hadm_id = cs.hadm_id
)
TO 'C:/Users/Lenovo/Desktop/Msc Project/data/features/charlson_scores.csv'
WITH CSV HEADER;