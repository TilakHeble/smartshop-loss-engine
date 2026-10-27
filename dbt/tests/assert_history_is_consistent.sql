-- History counts are internally consistent: earlier trips equal trip_seq - 1, store trips never
-- exceed all trips, a first trip has no history, and stores visited never exceed trips.
select * from {{ ref('fct_trip_history') }}
where prior_trips <> trip_seq - 1
   or prior_trips_this_store > prior_trips
   or (is_first_trip and (prior_avg_lines is not null or days_since_last_trip is not null))
   or (not is_first_trip and (prior_stores < 1 or prior_stores > prior_trips))
