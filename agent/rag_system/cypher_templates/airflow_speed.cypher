MATCH (a:Article)-[:CONSTRAINS]->(p:Parameter)-[:APPLIES_TO]->(loc:Location)
WHERE p.name CONTAINS '风速'
  AND (
    $location IS NULL
    OR loc.name CONTAINS $location
    OR $location CONTAINS loc.name
  )
WITH a, loc, p,
  CASE
    WHEN $airflow_speed IS NULL THEN NULL
    WHEN p.value_min IS NOT NULL AND $airflow_speed < p.value_min THEN false
    WHEN p.value_max IS NOT NULL AND $airflow_speed > p.value_max THEN false
    ELSE true
  END AS compliant
OPTIONAL MATCH (a)-[:SPECIFIES]->(req:Requirement)
OPTIONAL MATCH (req)-[:INVOLVES_FACILITY]->(fac:Facility)
WHERE $facility_type IS NULL
  OR fac.name IS NULL
  OR fac.name CONTAINS $facility_type
  OR $facility_type CONTAINS fac.name
  OR coalesce(req.content, '') CONTAINS $facility_type
WITH a, loc,
  collect(DISTINCT {
    name: p.name,
    value_min: p.value_min,
    value_max: p.value_max,
    unit: p.unit,
    observed: $airflow_speed,
    compliant: compliant
  }) AS constraints,
  collect(DISTINCT req.content) AS requirements,
  collect(DISTINCT fac.name) AS facilities
RETURN
  a.node_id AS node_id,
  a.name AS article_name,
  a.title AS article_title,
  a.content AS article_content,
  loc.name AS matched_location,
  constraints,
  requirements,
  facilities,
  'airflow_speed' AS template_id
ORDER BY article_name
LIMIT $limit
