SELECT careunit, COUNT(*) as frequency 
FROM mimiciv_hosp.transfers
GROUP BY careunit
ORDER BY count(*) DESC;