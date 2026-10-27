-- One row per member over the whole window. For reporting, not for features: features must use
-- fct_trip_history, which only sees the past. Demographics are deliberately NOT joined here.
with store_trips as (
    select member_id, store_id, count(*) as n
    from {{ ref('fct_trip') }}
    group by 1, 2
),

main_store as (
    select member_id, arg_max(store_id, n) as main_store_id, max(n) as main_store_trips
    from store_trips
    group by 1
)

select
    t.member_id,
    min(t.day)                                              as first_trip_day,
    max(t.day)                                              as last_trip_day,
    count(*)                                                as n_trips,
    count(distinct t.store_id)                              as n_stores,
    any_value(m.main_store_id)                              as main_store_id,
    any_value(m.main_store_trips) * 1.0 / count(*)          as main_store_share,
    avg(t.n_lines)                                          as avg_lines,
    avg(t.nectar_value)                                     as avg_trip_value,
    sum(t.high_risk_value) / sum(t.nectar_value)            as high_risk_share,
    sum(t.member_discount) / sum(t.shelf_value)             as nectar_saving_rate,
    t.member_id in (
        select member_id from {{ ref('stg_member_demographics') }}
    )                                                       as has_demographics
from {{ ref('fct_trip') }} t
join main_store m using (member_id)
group by t.member_id
