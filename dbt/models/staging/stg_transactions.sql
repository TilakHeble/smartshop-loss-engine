-- One row per basket line, with both prices a SmartShop screen shows.
--   shelf_value   normal price of the line (what a non-member pays)
--   nectar_value  Nectar price of the line (normal price less the member discount)
-- 3 of 2.3m lines record a positive "discount" (a data error): it is floored at zero.
select
    t.household_id                                          as member_id,
    t.basket_id,
    t.product_id,
    t.store_id,
    t.day,
    t.week_no,
    t.trans_time_hhmm,
    t.trans_time_hhmm // 100                                as hour_of_day,
    (t.trans_time_hhmm // 100) * 60 + t.trans_time_hhmm % 100  as minute_of_day,
    t.quantity,
    t.sales_value - t.coupon_match_disc                     as nectar_value,
    greatest(-t.retail_disc, 0)                             as member_discount,
    t.sales_value - t.coupon_match_disc
        + greatest(-t.retail_disc, 0)                       as shelf_value,
    -(t.coupon_disc + t.coupon_match_disc)                  as coupon_discount,
    -- A scanned item: a real product line, not fuel/till entries, returns, voids or zero lines.
    coalesce(p.is_scan_department, false)
        and t.quantity > 0 and t.sales_value > 0            as is_scan_line
from {{ source('raw', 'transactions') }} t
left join {{ ref('stg_products') }} p using (product_id)
where t.day <= {{ var('max_day') }}
