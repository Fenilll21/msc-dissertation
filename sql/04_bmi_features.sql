COPY (
WITH valid_bmi AS (
    
    SELECT
        subject_id,
        chartdate,
        CAST(result_value AS NUMERIC) AS bmi_value
    FROM mimiciv_hosp.omr
    WHERE result_name = 'BMI (kg/m2)'
    AND result_value ~ '^[0-9]+\.?[0-9]*$'
    AND CAST(result_value AS NUMERIC) BETWEEN 10 AND 100
),
ranked_bmi AS (
    
    SELECT
        a.hadm_id,
        a.subject_id,
        b.bmi_value,
        ROW_NUMBER() OVER (
            PARTITION BY a.hadm_id
            ORDER BY ABS(EXTRACT(EPOCH FROM (b.chartdate::timestamp - a.admittime)))
        ) AS rn
    FROM mimiciv_hosp.admissions a
    JOIN valid_bmi b ON a.subject_id = b.subject_id
)
SELECT
    a.hadm_id,
    a.subject_id,
    CASE
        WHEN r.bmi_value < 18.5              THEN 'Underweight'
        WHEN r.bmi_value BETWEEN 18.5 AND 24.9 THEN 'Normal'
        WHEN r.bmi_value BETWEEN 25.0 AND 29.9 THEN 'Overweight'
        WHEN r.bmi_value >= 30.0             THEN 'Obese'
        ELSE 'Unknown'
    END AS bmi_category
FROM mimiciv_hosp.admissions a
LEFT JOIN ranked_bmi r
    ON a.hadm_id = r.hadm_id
    AND r.rn = 1
)
TO 'C:/Users/Lenovo/Desktop/Msc Project/data/features/bmi_features.csv'
WITH CSV HEADER;