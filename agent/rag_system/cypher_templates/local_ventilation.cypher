MATCH (a:Article)
WHERE a.name IN ['第一百八十二条', '第一百八十三条', '第一百八十四条', '第一百九十七条']
OPTIONAL MATCH (a)-[:SPECIFIES]->(req:Requirement)
OPTIONAL MATCH (req)-[:INVOLVES_FACILITY]->(fac:Facility)
OPTIONAL MATCH (a)-[:CONSTRAINS]->(p:Parameter)
OPTIONAL MATCH (p)-[:APPLIES_TO]->(loc:Location)
WITH a,
  collect(DISTINCT req) AS req_nodes,
  collect(DISTINCT fac.name) AS facilities,
  collect(DISTINCT p) AS param_nodes,
  collect(DISTINCT loc.name) AS locations
WHERE $facility_type IS NULL
  OR a.title CONTAINS $facility_type
  OR any(f IN facilities WHERE f IS NOT NULL AND (f CONTAINS $facility_type OR $facility_type CONTAINS f))
  OR any(r IN req_nodes WHERE coalesce(r.content, '') CONTAINS $facility_type)
WITH a, req_nodes, facilities, locations,
  [p IN param_nodes | {
    name: p.name,
    value_min: p.value_min,
    value_max: p.value_max,
    unit: p.unit,
    observed_distance_to_return_air_m: $distance_to_return_air_m,
    compliant: CASE
      WHEN $distance_to_return_air_m IS NULL THEN NULL
      WHEN p.name CONTAINS '距掘进巷道回风口最小距离' AND p.value_min IS NOT NULL AND $distance_to_return_air_m < p.value_min THEN false
      ELSE true
    END
  }] AS constraints
RETURN
  a.node_id AS node_id,
  a.name AS article_name,
  a.title AS article_title,
  a.content AS article_content,
  [r IN req_nodes | r.content] AS requirements,
  facilities,
  locations,
  constraints,
  {
    has_wind_power_lock: $has_wind_power_lock,
    has_methane_lock: $has_methane_lock,
    has_backup_fan: $has_backup_fan,
    air_duct_material: $air_duct_material,
    air_duct_status: $air_duct_status
  } AS observed,
  'local_ventilation' AS template_id
ORDER BY article_name
LIMIT $limit
