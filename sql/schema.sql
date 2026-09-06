CREATE INDEX IF NOT EXISTS
idx_credit_applicants_target
ON credit_applicants(TARGET);

CREATE INDEX IF NOT EXISTS
idx_credit_applicants_risk_band
ON credit_applicants(RISK_BAND);

CREATE INDEX IF NOT EXISTS
idx_credit_applicants_income_type
ON credit_applicants(NAME_INCOME_TYPE);