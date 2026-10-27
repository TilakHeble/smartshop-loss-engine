-- Every value in the high-risk seed must exist in the product table (catches typos that would
-- silently drop a category).
select r.* from {{ ref('high_risk_categories') }} r
where not exists (
    select 1 from {{ source('raw', 'products') }} p
    where (r.match_on = 'commodity' and p.commodity = r.value)
       or (r.match_on = 'department' and p.department = r.value)
)
