-- What the system knows about the shopper at the moment a session starts: one row per trip,
-- built ONLY from the member's earlier trips (window frames end at 1 PRECEDING).
-- Nectar links trips across stores, so history is per member, not per store.
-- Rescan history (past rescans and their outcomes) is added once rescans are simulated (week 3).
with t as (
    select
        *,
        row_number() over (
            partition by member_id, store_id order by trip_seq
        )                                                   as store_trip_seq
    from {{ ref('fct_trip') }}
),

w as (
    select
        basket_id,
        member_id,
        store_id,
        day,
        trip_seq,
        trip_seq - 1                                        as prior_trips,
        store_trip_seq - 1                                  as prior_trips_this_store,
        coalesce(sum((store_trip_seq = 1)::int) over past, 0)  as prior_stores,
        day - lag(day) over (partition by member_id order by trip_seq)  as days_since_last_trip,
        day - first_value(day) over (partition by member_id order by trip_seq)  as tenure_days,
        avg(n_lines) over past                              as prior_avg_lines,
        avg(nectar_value) over past                         as prior_avg_value,
        sum(high_risk_value) over past
            / nullif(sum(nectar_value) over past, 0)        as prior_high_risk_share,
        avg(top3_value_share) over past                     as prior_avg_top3_share
    from t
    window past as (
        partition by member_id order by trip_seq
        rows between unbounded preceding and 1 preceding
    )
)

select
    *,
    prior_trips = 0                                         as is_first_trip,
    prior_trips_this_store * 1.0 / nullif(prior_trips, 0)   as prior_share_this_store
from w
