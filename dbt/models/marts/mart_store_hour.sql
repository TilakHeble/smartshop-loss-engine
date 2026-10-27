-- Store x hour-of-day demand: average sessions and item lines arriving per open day. Week 6 turns
-- this into rescan capacity per hour (how many rescans colleagues can do without a queue).
with s as (
    select store_id, active_days from {{ ref('dim_store') }}
)

select
    t.store_id,
    t.hour_of_day,
    count(*)                                                as n_trips,
    count(*) * 1.0 / any_value(s.active_days)               as trips_per_day,
    sum(t.n_lines) * 1.0 / any_value(s.active_days)         as lines_per_day,
    avg(t.n_lines)                                          as avg_lines
from {{ ref('fct_trip') }} t
join s using (store_id)
group by 1, 2
