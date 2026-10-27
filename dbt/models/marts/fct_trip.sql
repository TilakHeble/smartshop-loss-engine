-- One row per SmartShop session (a shopping trip), built from its scanned lines.
-- trip_seq orders each member's trips in time; it is the backbone of point-in-time history.
with trips as (
    select
        basket_id,
        any_value(member_id)                                as member_id,
        any_value(store_id)                                 as store_id,
        min(day)                                            as day,
        min(week_no)                                        as week_no,
        min(hour_of_day)                                    as hour_of_day,
        min(minute_of_day)                                  as checkout_minute,
        count(*)                                            as n_lines,
        sum(quantity)                                       as n_units,
        count(distinct department)                          as n_departments,
        sum(nectar_value)                                   as nectar_value,
        sum(shelf_value)                                    as shelf_value,
        sum(member_discount)                                as member_discount,
        count(*) filter (where has_nectar_price)            as n_nectar_price_lines,
        count(*) filter (where is_high_risk)                as n_high_risk_lines,
        coalesce(sum(nectar_value) filter (where is_high_risk), 0)  as high_risk_value,
        count(*) filter (where is_multi_buy)                as n_multi_buy_lines,
        max(nectar_value)                                   as max_line_value,
        sum(nectar_value) filter (where value_rank <= 3)    as top3_value
    from {{ ref('fct_scan_line') }}
    group by basket_id
)

select
    *,
    high_risk_value / nectar_value                          as high_risk_share,
    top3_value / nectar_value                               as top3_value_share,
    hour_of_day in ({{ var('peak_hours') | join(', ') }})   as is_peak_hour,
    row_number() over (
        partition by member_id order by day, checkout_minute, basket_id
    )                                                       as trip_seq
from trips
