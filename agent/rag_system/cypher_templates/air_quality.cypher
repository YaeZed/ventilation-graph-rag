MATCH (a:Article)-[:CONSTRAINS]->(p:Parameter)
OPTIONAL MATCH (p)-[:APPLIES_TO]->(loc:Location)
WITH a, p, loc,
  CASE
    WHEN p.name CONTAINS '氧气' THEN $oxygen_concentration
    WHEN p.name CONTAINS '二氧化碳' THEN $carbon_dioxide_concentration
    WHEN p.name CONTAINS '甲烷' THEN $methane_concentration
    WHEN p.name CONTAINS '一氧化碳' THEN $carbon_monoxide_concentration
    WHEN p.name CONTAINS '氧化氮' THEN $nitrogen_oxide_concentration
    WHEN p.name CONTAINS '二氧化硫' THEN $sulfur_dioxide_concentration
    WHEN p.name CONTAINS '硫化氢' THEN $hydrogen_sulfide_concentration
    WHEN p.name CONTAINS '氨' THEN $ammonia_concentration
    ELSE NULL
  END AS observed
WHERE observed IS NOT NULL
  AND (
    $location IS NULL
    OR coalesce(loc.name, '') = ''
    OR loc.name = '通用'
    OR loc.name CONTAINS $location
    OR $location CONTAINS loc.name
  )
WITH a, p, loc, observed,
  CASE
    WHEN p.value_min IS NOT NULL AND observed < p.value_min THEN false
    WHEN p.value_max IS NOT NULL AND observed > p.value_max THEN false
    ELSE true
  END AS compliant
OPTIONAL MATCH (a)-[:SPECIFIES]->(req:Requirement)
RETURN
  a.node_id AS node_id,
  a.name AS article_name,
  a.title AS article_title,
  a.content AS article_content,
  loc.name AS matched_location,
  collect(DISTINCT {
    name: p.name,
    value_min: p.value_min,
    value_max: p.value_max,
    unit: p.unit,
    observed: observed,
    compliant: compliant
  }) AS constraints,
  collect(DISTINCT req.content) AS requirements,
  'air_quality' AS template_id
ORDER BY article_name
LIMIT $limit
