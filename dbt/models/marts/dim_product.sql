-- One row per product that was scanned at least once: its risk group and price profile.
-- The unit prices value an unscanned item at the price the shopper would have paid.
select
    p.product_id,
    p.department,
    p.commodity,
    p.sub_commodity,
    p.brand_type,
    p.pack_size,
    p.risk_group,
    p.is_high_risk,
    count(*)                                                as n_lines,
    count(distinct l.basket_id)                             as n_trips,
    sum(l.quantity)                                         as n_units,
    sum(l.shelf_value) / sum(l.quantity)                    as shelf_unit_price,
    sum(l.nectar_value) / sum(l.quantity)                   as nectar_unit_price,
    avg(l.has_nectar_price::int)                            as nectar_price_share,
    sum(l.member_discount) / sum(l.shelf_value)             as avg_discount_rate
from {{ ref('fct_scan_line') }} l
join {{ ref('stg_products') }} p using (product_id)
group by all
