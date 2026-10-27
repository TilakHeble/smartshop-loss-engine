-- One row per item line a SmartShop shopper scans (weeks 17+). This is the "scan event" table at
-- line grain: week 3's simulator expands it into timed, per-unit scans and adds what was missed.
select
    t.basket_id,
    t.product_id,
    t.member_id,
    t.store_id,
    t.day,
    t.week_no,
    t.hour_of_day,
    t.minute_of_day,
    p.department,
    p.commodity,
    p.risk_group,
    p.is_high_risk,
    t.quantity,
    t.quantity > 1                                          as is_multi_buy,
    t.shelf_value,
    t.nectar_value,
    t.member_discount,
    t.member_discount > 0                                   as has_nectar_price,
    t.nectar_value / t.quantity                             as nectar_unit_price,
    -- 1 = the most valuable line in the basket (what a targeted rescan would check first)
    row_number() over (
        partition by t.basket_id order by t.nectar_value desc, t.product_id
    )                                                       as value_rank
from {{ ref('stg_transactions') }} t
join {{ ref('stg_products') }} p using (product_id)
where t.is_scan_line
  and t.week_no >= {{ var('first_stable_week') }}
