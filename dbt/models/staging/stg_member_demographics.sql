-- Demographics exist for 801 of 2,500 members. They are kept in their own model and used ONLY
-- for the fairness check: no feature or decision model may join this table.
select
    household_id            as member_id,
    age_group,
    income_level,
    household_size,
    kids
from {{ source('raw', 'households') }}
