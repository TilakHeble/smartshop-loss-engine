-- Trip totals equal the sum of their scanned lines, and all scanned lines sum to the staging
-- scan lines in the analysis window (to the cent).
with l as (select sum(nectar_value) v, sum(shelf_value) s, count(*) n from {{ ref('fct_scan_line') }}),
t as (select sum(nectar_value) v, sum(shelf_value) s, sum(n_lines) n from {{ ref('fct_trip') }}),
st as (
    select sum(nectar_value) v, count(*) n from {{ ref('stg_transactions') }}
    where is_scan_line and week_no >= {{ var('first_stable_week') }}
)
select * from l, t, st
where abs(l.v - t.v) > 0.01 or abs(l.s - t.s) > 0.01 or l.n <> t.n
   or abs(l.v - st.v) > 0.01 or l.n <> st.n
