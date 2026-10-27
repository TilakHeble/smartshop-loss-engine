-- One row per store: size and how peaked its demand is (context for rescan capacity).
select
    store_id,
    count(*)                                                as n_trips,
    count(distinct member_id)                               as n_members,
    count(distinct day)                                     as active_days,
    count(*) * 1.0 / count(distinct day)                    as trips_per_active_day,
    avg(n_lines)                                            as avg_lines,
    sum(nectar_value)                                       as nectar_value,
    avg(is_peak_hour::int)                                  as peak_share,
    mode(hour_of_day)                                       as busiest_hour
from {{ ref('fct_trip') }}
group by store_id
