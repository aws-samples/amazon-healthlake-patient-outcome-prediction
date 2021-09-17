CREATE OR REPLACE VIEW pop_pat_enc AS 
SELECT DISTINCT
  patient_id
, pat_ref_id
, enc_ref_id
, (CASE WHEN (age >= 90) THEN 90 ELSE age END) age
, gender
, if_married
, encounter_start
, encounter_end
FROM
  (
   SELECT DISTINCT
     p.identifier[1].value patient_id
   , "split"(enc.subject.reference, '/')[2] pat_ref_id
   , (CASE WHEN (p.gender = 'male') THEN 1 ELSE 0 END) gender
   , (CASE WHEN (p.maritalstatus.coding[1].display = 'Married') THEN 1 ELSE 0 END) if_married
   , ("year"("from_iso8601_timestamp"(enc.period."end")) - "year"("from_iso8601_timestamp"(p.birthdate))) age
   , "date"("from_iso8601_timestamp"(enc.period.start)) encounter_start
   , "date"("from_iso8601_timestamp"(enc.period."end")) encounter_end
   , enc.id enc_ref_id
   FROM
     (healthlake_synthea.encounter enc
   LEFT JOIN healthlake_synthea.patient p ON ("split"(enc.subject.reference, '/')[2] = p.id))
) 

--------------------------------------------------

CREATE OR REPLACE VIEW pop_cond AS 
SELECT DISTINCT
  pat_ref_id
, a.enc_ref_id
, covid_date
, "array_agg"(cond_cd) OVER (PARTITION BY pat_ref_id, a.enc_ref_id) cond_cd
FROM
  ((
   SELECT
     enc.id enc_ref_id
   , "split"(subject.reference, '/')[2] pat_ref_id
   FROM
     healthlake_synthea.encounter enc
)  a
LEFT JOIN (
   SELECT
     "split"(con.encounter.reference, '/')[2] enc_ref_id
   , con.code.coding[1].code cond_cd
   , (CASE WHEN (con.code.coding[1].code = '840539006') THEN "date"("from_iso8601_timestamp"(con.recordeddate)) ELSE null END) covid_date
   FROM
     healthlake_synthea.condition con
)  b ON (a.enc_ref_id = b.enc_ref_id))


--------------------------------------------------

CREATE OR REPLACE VIEW pop_proc AS 
SELECT DISTINCT
  pat_ref_id
, enc_ref_id
, "array_agg"(proc_cd) OVER (PARTITION BY pat_ref_id, enc_ref_id) proc_cd
FROM
  ((
   SELECT
     enc.id enc_ref_id
   , "split"(subject.reference, '/')[2] pat_ref_id
   FROM
     healthlake_synthea.encounter enc
)  a
LEFT JOIN (
   SELECT
     "split"(proc.encounter.reference, '/')[2] enc_ref_id2
   , proc.code.coding[1].code proc_cd
   FROM
     healthlake_synthea.procedure proc
)  b ON (a.enc_ref_id = b.enc_ref_id2))


--------------------------------------------------

CREATE OR REPLACE VIEW pop_med AS 
SELECT DISTINCT
  a.pat_ref_id
, a.enc_ref_id
, "array_agg"(drug_nm) OVER (PARTITION BY a.pat_ref_id, a.enc_ref_id) drug_nm
, "array_agg"(is_rxnorm) OVER (PARTITION BY a.pat_ref_id, a.enc_ref_id) is_rxnorm
, "array_agg"(rxnorm) OVER (PARTITION BY a.pat_ref_id, a.enc_ref_id) rxnorm
FROM
  ((
   SELECT
     enc.id enc_ref_id
   , "split"(subject.reference, '/')[2] pat_ref_id
   FROM
     healthlake_synthea.encounter enc
)  a
LEFT JOIN (
   SELECT
     med.medicationcodeableconcept.coding[1].code rxnorm
   , (CASE WHEN (med.medicationcodeableconcept.coding[1]."system" = 'http://www.nlm.nih.gov/research/umls/rxnorm') THEN 1 ELSE 0 END) is_rxnorm
   , med.medicationcodeableconcept."text" drug_nm
   , "split"(med.subject.reference, '/')[2] pat_ref_id
   , "split"(med.context.reference, '/')[2] enc_ref_id
   FROM
     healthlake_synthea.medicationadministration med
)  b ON ((a.enc_ref_id = b.enc_ref_id) AND (a.pat_ref_id = b.pat_ref_id)))


--------------------------------------------------

CREATE OR REPLACE VIEW pop_obs AS 
SELECT DISTINCT
  a.pat_ref_id
, a.enc_ref_id
, "array_agg"(loinc_cd) OVER (PARTITION BY a.pat_ref_id, a.enc_ref_id) loinc_cd
, "array_agg"(lab_nm) OVER (PARTITION BY a.pat_ref_id, a.enc_ref_id) obs_nm
, "array_agg"(CAST(avg_lab_val AS varchar)) OVER (PARTITION BY a.pat_ref_id, a.enc_ref_id) obs_val
FROM
  ((
   SELECT
     enc.id enc_ref_id
   , "split"(subject.reference, '/')[2] pat_ref_id
   FROM
     healthlake_synthea.encounter enc
)  a
LEFT JOIN (
   SELECT DISTINCT
     pat_ref_id
   , enc_ref_id
   , lab_nm
   , loinc_cd
   , "round"("avg"(lab_val) OVER (PARTITION BY pat_ref_id, enc_ref_id, lab_nm, loinc_cd), 1) avg_lab_val
   FROM
     (
      SELECT
        "split"(obs.subject.reference, '/')[2] pat_ref_id
      , "split"(obs.encounter.reference, '/')[2] enc_ref_id
      , "replace"("replace"(obs.code."text", ',', ''), ' ', '') lab_nm
      , obs.code.coding[1].code loinc_cd
      , obs.valuequantity.value lab_val
      FROM
        healthlake_synthea.observation obs
   ) 
)  b ON ((a.enc_ref_id = b.enc_ref_id) AND (a.pat_ref_id = b.pat_ref_id)))



--------------------------------------------------

CREATE OR REPLACE VIEW pop_main AS 
SELECT DISTINCT
  patient_id
, pop_pat_enc.pat_ref_id
, pop_pat_enc.enc_ref_id
, age
, gender
, if_married
, encounter_start
, encounter_end
, cond_cd
, proc_cd
, drug_nm
, rxnorm
, is_rxnorm
, loinc_cd
, obs_nm
, obs_val
, (CASE WHEN (("date_diff"('day', encounter_end, covid_date) >= 0) AND ("date_diff"('day', encounter_end, covid_date) <= 90)) THEN 1 ELSE 0 END) covid90
FROM
  ((((pop_pat_enc
INNER JOIN pop_cond ON (pop_pat_enc.enc_ref_id = pop_cond.enc_ref_id))
LEFT JOIN pop_proc ON (pop_pat_enc.enc_ref_id = pop_proc.enc_ref_id))
INNER JOIN pop_med ON (pop_pat_enc.enc_ref_id = pop_med.enc_ref_id))
INNER JOIN pop_obs ON (pop_pat_enc.enc_ref_id = pop_obs.enc_ref_id))
--------------------------------------------------