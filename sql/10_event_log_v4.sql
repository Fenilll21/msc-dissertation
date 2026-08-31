COPY (
    -- Hospital Admission anchor from admissions table
    SELECT
        a.hadm_id AS case_id,
        a.admittime AS event_time,
        'Hospital Admission' AS activity
    FROM mimiciv_hosp.admissions a

    UNION ALL

    SELECT
        t.hadm_id AS case_id,
        t.intime AS event_time,
        CASE
            -- Emergency Department
            WHEN t.careunit IN (
                'Emergency Department',
                'Emergency Department Observation'
            ) AND t.eventtype IN ('admit', 'ED') THEN 'Admission to Emergency Department'
            WHEN t.careunit IN (
                'Emergency Department',
                'Emergency Department Observation'
            ) AND t.eventtype = 'transfer'      THEN 'Transfer to Emergency Department'

            -- ICU
            WHEN t.careunit IN (
                'Medical Intensive Care Unit (MICU)',
                'Cardiac Vascular Intensive Care Unit (CVICU)',
                'Medical/Surgical Intensive Care Unit (MICU/SICU)',
                'Surgical Intensive Care Unit (SICU)',
                'Trauma SICU (TSICU)',
                'Coronary Care Unit (CCU)',
                'Neuro Surgical Intensive Care Unit (Neuro SICU)',
                'Intensive Care Unit (ICU)'
            ) AND t.eventtype = 'admit'         THEN 'Admission to ICU'
            WHEN t.careunit IN (
                'Medical Intensive Care Unit (MICU)',
                'Cardiac Vascular Intensive Care Unit (CVICU)',
                'Medical/Surgical Intensive Care Unit (MICU/SICU)',
                'Surgical Intensive Care Unit (SICU)',
                'Trauma SICU (TSICU)',
                'Coronary Care Unit (CCU)',
                'Neuro Surgical Intensive Care Unit (Neuro SICU)',
                'Intensive Care Unit (ICU)'
            ) AND t.eventtype = 'transfer'      THEN 'Transfer to ICU'

            -- Surgery Ward
            WHEN t.careunit IN (
                'Surgery', 'Surgery/Trauma', 'Cardiac Surgery',
                'Thoracic Surgery',
                'Surgery/Pancreatic/Biliary/Bariatric',
                'Med/Surg/Trauma'
            ) AND t.eventtype = 'admit'         THEN 'Admission to Surgery Ward'
            WHEN t.careunit IN (
                'Surgery', 'Surgery/Trauma', 'Cardiac Surgery',
                'Thoracic Surgery',
                'Surgery/Pancreatic/Biliary/Bariatric',
                'Med/Surg/Trauma'
            ) AND t.eventtype = 'transfer'      THEN 'Transfer to Surgery Ward'

            -- Stepdown/Intermediate
            WHEN t.careunit IN (
                'PACU',
                'Hematology/Oncology Intermediate',
                'Medicine/Cardiology Intermediate',
                'Cardiology Surgery Intermediate',
                'Neuro Intermediate', 'Neuro Stepdown',
                'Surgery/Vascular/Intermediate',
                'Surgical Intermediate'
            ) AND t.eventtype = 'admit'         THEN 'Admission to Stepdown'
            WHEN t.careunit IN (
                'PACU',
                'Hematology/Oncology Intermediate',
                'Medicine/Cardiology Intermediate',
                'Cardiology Surgery Intermediate',
                'Neuro Intermediate', 'Neuro Stepdown',
                'Surgery/Vascular/Intermediate',
                'Surgical Intermediate'
            ) AND t.eventtype = 'transfer'      THEN 'Transfer to Stepdown'

            -- Medical Ward
            WHEN t.careunit IN (
                'Medicine', 'Med/Surg', 'Medicine/Cardiology',
                'Neurology', 'Hematology/Oncology', 'Vascular',
                'Psychiatry', 'Cardiology', 'Oncology', 'Transplant',
                'Med/Surg/GYN', 'Observation', 'Discharge Lounge'
            ) AND t.eventtype = 'admit'         THEN 'Admission to Medical Ward'
            WHEN t.careunit IN (
                'Medicine', 'Med/Surg', 'Medicine/Cardiology',
                'Neurology', 'Hematology/Oncology', 'Vascular',
                'Psychiatry', 'Cardiology', 'Oncology', 'Transplant',
                'Med/Surg/GYN', 'Observation', 'Discharge Lounge'
            ) AND t.eventtype = 'transfer'      THEN 'Transfer to Medical Ward'

            ELSE NULL
        END AS activity
    FROM mimiciv_hosp.transfers t
    WHERE t.hadm_id IS NOT NULL
    AND t.eventtype != 'discharge'
)
TO 'C:/Users/Lenovo/Desktop/Msc Project/data/event_log/mimic_transfers_v4.csv'
WITH CSV HEADER;