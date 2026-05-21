MATCH (a:Article)-[:SPECIFIES]->(req:Requirement)
OPTIONAL MATCH (req)-[:INVOLVES_FACILITY]->(fac:Facility)
WHERE (
    $facility_type IS NULL
    OR a.title CONTAINS $facility_type
    OR coalesce(req.content, '') CONTAINS $facility_type
    OR fac.name CONTAINS $facility_type
    OR $facility_type CONTAINS fac.name
  )
  AND (
    $location IS NULL
    OR coalesce(req.content, '') CONTAINS $location
    OR a.title CONTAINS $location
  )
WITH a,
  collect(DISTINCT req.content) AS requirements,
  collect(DISTINCT fac.name) AS facilities
OPTIONAL MATCH (a)-[:CONSTRAINS]->(p:Parameter)
OPTIONAL MATCH (p)-[:APPLIES_TO]->(loc:Location)
RETURN
  a.node_id AS node_id,
  a.name AS article_name,
  a.title AS article_title,
  a.content AS article_content,
  requirements,
  facilities,
  collect(DISTINCT {
    name: p.name,
    value_min: p.value_min,
    value_max: p.value_max,
    unit: p.unit,
    location: loc.name
  }) AS constraints,
  {
    door_count: $door_count,
    has_reverse_door: $has_reverse_door,
    is_interlocked: $is_interlocked,
    inclination_deg: $inclination_deg,
    is_sealed: $is_sealed
  } AS observed,
  'ventilation_facility' AS template_id
ORDER BY article_name
LIMIT $limit
