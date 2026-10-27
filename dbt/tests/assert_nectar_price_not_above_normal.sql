-- The Nectar price of a line can never be above its normal price.
select * from {{ ref('fct_scan_line') }}
where nectar_value > shelf_value + 1e-9
