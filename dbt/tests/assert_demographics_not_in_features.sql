-- Guardrail: models used for features and decisions must not carry demographic columns.
select table_name, column_name from information_schema.columns
where table_schema = 'marts'
  and table_name in ('fct_scan_line', 'fct_trip', 'fct_trip_history', 'dim_member', 'dim_product')
  and column_name in ('age_group', 'income_level', 'household_size', 'kids',
                      'marital_group', 'homeowner', 'hh_composition_group')
