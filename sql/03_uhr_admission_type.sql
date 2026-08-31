COPY (
WITH classified_admissions AS (
    -- Step 1: Classify every admission as emergency or elective
    SELECT
        hadm_id,
        subject_id,
        admittime,
        dischtime,
        CASE
            WHEN admission_type IN (
                'EW EMER.',
                'EU OBSERVATION',
                'OBSERVATION ADMIT',
                'URGENT',
                'DIRECT EMER.',
                'DIRECT OBSERVATION'
            ) THEN 'emergency'
            ELSE 'elective'
        END AS admission_type,
        CASE
            WHEN admission_type IN (
                'EW EMER.',
                'EU OBSERVATION',
                'OBSERVATION ADMIT',
                'URGENT',
                'DIRECT EMER.',
                'DIRECT OBSERVATION'
            ) THEN 1
            ELSE 0
        END AS is_emergency
    FROM mimiciv_hosp.admissions
),

with_next AS (
    -- Step 2: For each admission, look at the next admission for same patient
    SELECT
        hadm_id,
        subject_id,
        admittime,
        dischtime,
        admission_type,
        is_emergency,
        LEAD(admittime) OVER (
            PARTITION BY subject_id ORDER BY admittime
        ) AS next_admittime,
        LEAD(is_emergency) OVER (
            PARTITION BY subject_id ORDER BY admittime
        ) AS next_is_emergency
    FROM classified_admissions
),

flagged AS (
    -- Step 3: Flag whether this admission was followed by 30-day emergency readmission
    SELECT
        hadm_id,
        subject_id,
        admittime,
        admission_type,
        CASE
            WHEN next_admittime IS NOT NULL
                AND next_is_emergency = 1
                AND EXTRACT(EPOCH FROM (next_admittime - dischtime)) / 86400 <= 30
            THEN 1
            ELSE 0
        END AS was_followed_by_uhr
    FROM with_next
)

-- Step 4: For each admission, count prior was_followed_by_uhr = 1 events
SELECT
    a.hadm_id,
    a.subject_id,
    a.admission_type,
    COUNT(prior.hadm_id) FILTER (WHERE prior.was_followed_by_uhr = 1) AS uhr_count
FROM flagged a
LEFT JOIN flagged prior
    ON a.subject_id = prior.subject_id
    AND prior.admittime < a.admittime
GROUP BY a.hadm_id, a.subject_id, a.admission_type
)
TO 'C:/Users/Lenovo/Desktop/Msc Project/data/features/uhr_admission_type.csv'
WITH CSV HEADER;