-- One row per product, with its high-risk group (seed: high_risk_categories) and whether it is
-- something a shopper scans. A commodity match wins over a department match.
with risk as (
    select * from {{ ref('high_risk_categories') }}
)

select
    p.product_id,
    p.manufacturer_id,
    p.department,
    p.commodity,
    p.sub_commodity,
    p.brand_type,
    p.pack_size,
    coalesce(rc.risk_group, rd.risk_group)              as risk_group,
    coalesce(rc.risk_group, rd.risk_group) is not null  as is_high_risk,
    p.department not in (select department from {{ ref('non_scan_departments') }})
        and coalesce(p.department, '') <> ''            as is_scan_department
from {{ source('raw', 'products') }} p
left join risk rc on rc.match_on = 'commodity' and rc.value = p.commodity
left join risk rd on rd.match_on = 'department' and rd.value = p.department
