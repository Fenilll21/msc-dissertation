COPY (
    SELECT hadm_id, outtime AS prediction_point
    FROM (
        SELECT hadm_id, outtime,
               ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) AS rn
        FROM mimiciv_hosp.transfers
        WHERE careunit IN (
            'Surgery', 'Surgery/Trauma', 'Cardiac Surgery',
            'Thoracic Surgery',
            'Surgery/Pancreatic/Biliary/Bariatric',
            'Med/Surg/Trauma'
        )
    ) t
    WHERE rn = 1
)
TO 'C:/Users/Lenovo/Desktop/Msc Project/data/features/sw_prediction_points.csv'
WITH CSV HEADER;