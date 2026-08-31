COPY (
WITH classified AS (
    SELECT
        hadm_id,
        CASE
            -- ICD-9 classification by numeric range
            WHEN icd_version = 9 AND icd_code ~ '^[0-9]{3}' THEN
                CASE
                    WHEN CAST(SUBSTRING(icd_code,1,3) AS INT) BETWEEN 1   AND 139 THEN 1
                    WHEN CAST(SUBSTRING(icd_code,1,3) AS INT) BETWEEN 140 AND 239 THEN 2
                    WHEN CAST(SUBSTRING(icd_code,1,3) AS INT) BETWEEN 240 AND 279 THEN 3
                    WHEN CAST(SUBSTRING(icd_code,1,3) AS INT) BETWEEN 280 AND 289 THEN 4
                    WHEN CAST(SUBSTRING(icd_code,1,3) AS INT) BETWEEN 290 AND 319 THEN 5
                    WHEN CAST(SUBSTRING(icd_code,1,3) AS INT) BETWEEN 320 AND 389 THEN 6
                    WHEN CAST(SUBSTRING(icd_code,1,3) AS INT) BETWEEN 390 AND 459 THEN 7
                    WHEN CAST(SUBSTRING(icd_code,1,3) AS INT) BETWEEN 460 AND 519 THEN 8
                    WHEN CAST(SUBSTRING(icd_code,1,3) AS INT) BETWEEN 520 AND 579 THEN 9
                    WHEN CAST(SUBSTRING(icd_code,1,3) AS INT) BETWEEN 580 AND 629 THEN 10
                    WHEN CAST(SUBSTRING(icd_code,1,3) AS INT) BETWEEN 630 AND 679 THEN 11
                    WHEN CAST(SUBSTRING(icd_code,1,3) AS INT) BETWEEN 680 AND 709 THEN 12
                    WHEN CAST(SUBSTRING(icd_code,1,3) AS INT) BETWEEN 710 AND 739 THEN 13
                    WHEN CAST(SUBSTRING(icd_code,1,3) AS INT) BETWEEN 740 AND 759 THEN 14
                    WHEN CAST(SUBSTRING(icd_code,1,3) AS INT) BETWEEN 760 AND 779 THEN 15
                    WHEN CAST(SUBSTRING(icd_code,1,3) AS INT) BETWEEN 780 AND 799 THEN 16
                    WHEN CAST(SUBSTRING(icd_code,1,3) AS INT) BETWEEN 800 AND 999 THEN 17
                    ELSE NULL
                END
            WHEN icd_version = 9 AND icd_code ~ '^[VvEe]' THEN 18

            -- ICD-10 classification by letter prefix
            WHEN icd_version = 10 THEN
                CASE
                    WHEN icd_code ~ '^[AB]'       THEN 1   -- Infectious
                    WHEN icd_code ~ '^C'           THEN 2   -- Neoplasms
                    WHEN icd_code ~ '^D[0-4]'      THEN 2   -- Neoplasms (D00-D49)
                    WHEN icd_code ~ '^D[5-8]'      THEN 4   -- Blood (D50-D89)
                    WHEN icd_code ~ '^E'           THEN 3   -- Endocrine
                    WHEN icd_code ~ '^F'           THEN 5   -- Mental
                    WHEN icd_code ~ '^[GH]'        THEN 6   -- Nervous/sense organs
                    WHEN icd_code ~ '^I'           THEN 7   -- Circulatory
                    WHEN icd_code ~ '^J'           THEN 8   -- Respiratory
                    WHEN icd_code ~ '^K'           THEN 9   -- Digestive
                    WHEN icd_code ~ '^N'           THEN 10  -- Genitourinary
                    WHEN icd_code ~ '^O'           THEN 11  -- Pregnancy
                    WHEN icd_code ~ '^L'           THEN 12  -- Skin
                    WHEN icd_code ~ '^M'           THEN 13  -- Musculoskeletal
                    WHEN icd_code ~ '^Q'           THEN 14  -- Congenital
                    WHEN icd_code ~ '^P'           THEN 15  -- Perinatal
                    WHEN icd_code ~ '^R'           THEN 16  -- Symptoms/signs
                    WHEN icd_code ~ '^[ST]'        THEN 17  -- Injury/poisoning
                    WHEN icd_code ~ '^[UVWXYZ]'   THEN 18  -- Supplementary
                    ELSE NULL
                END
            ELSE NULL
        END AS chapter
    FROM mimiciv_hosp.diagnoses_icd
    WHERE hadm_id IS NOT NULL
)
SELECT
    a.hadm_id,
    a.subject_id,
    COUNT(CASE WHEN c.chapter = 1  THEN 1 END) AS icd_ch1_infectious,
    COUNT(CASE WHEN c.chapter = 2  THEN 1 END) AS icd_ch2_neoplasms,
    COUNT(CASE WHEN c.chapter = 3  THEN 1 END) AS icd_ch3_endocrine,
    COUNT(CASE WHEN c.chapter = 4  THEN 1 END) AS icd_ch4_blood,
    COUNT(CASE WHEN c.chapter = 5  THEN 1 END) AS icd_ch5_mental,
    COUNT(CASE WHEN c.chapter = 6  THEN 1 END) AS icd_ch6_nervous,
    COUNT(CASE WHEN c.chapter = 7  THEN 1 END) AS icd_ch7_circulatory,
    COUNT(CASE WHEN c.chapter = 8  THEN 1 END) AS icd_ch8_respiratory,
    COUNT(CASE WHEN c.chapter = 9  THEN 1 END) AS icd_ch9_digestive,
    COUNT(CASE WHEN c.chapter = 10 THEN 1 END) AS icd_ch10_genitourinary,
    COUNT(CASE WHEN c.chapter = 11 THEN 1 END) AS icd_ch11_pregnancy,
    COUNT(CASE WHEN c.chapter = 12 THEN 1 END) AS icd_ch12_skin,
    COUNT(CASE WHEN c.chapter = 13 THEN 1 END) AS icd_ch13_musculoskeletal,
    COUNT(CASE WHEN c.chapter = 14 THEN 1 END) AS icd_ch14_congenital,
    COUNT(CASE WHEN c.chapter = 15 THEN 1 END) AS icd_ch15_perinatal,
    COUNT(CASE WHEN c.chapter = 16 THEN 1 END) AS icd_ch16_symptoms,
    COUNT(CASE WHEN c.chapter = 17 THEN 1 END) AS icd_ch17_injury,
    COUNT(CASE WHEN c.chapter = 18 THEN 1 END) AS icd_ch18_supplementary
FROM mimiciv_hosp.admissions a
LEFT JOIN classified c ON a.hadm_id = c.hadm_id
GROUP BY a.hadm_id, a.subject_id
)
TO 'C:/Users/Lenovo/Desktop/Msc Project/data/features/icd_chapter_counts.csv'
WITH CSV HEADER;