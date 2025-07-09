CREATE OR REPLACE TABLE `patent-infos.Honeywell_patents.joined_patent_info` AS
-- Step 1: Get list of patent IDs of interest
WITH patent_list AS (
  SELECT patent_id
  FROM `patent-infos.Honeywell_patents.patent_list`
),
-- Step 2: Filter the massive publications table to only matching patents
filtered_publications AS (
  SELECT
    publication_number,
    kind_code,
    family_id,
    title_localized,
    abstract_localized,
    claims_localized,
    description_localized,
    publication_date,
    SPLIT(publication_number, '-')[OFFSET(1)] AS base_pub_number
  FROM
    `patents-public-data.patents.publications*` 
  WHERE 
    country_code = 'US'
    AND SPLIT(publication_number, '-')[OFFSET(1)] IN (
      SELECT patent_id FROM patent_list
    )
),
-- Step 3: Get the latest publication per patent_id
latest_by_patent AS (
  SELECT
    base_pub_number,
    ARRAY_AGG(STRUCT(
      kind_code,
      family_id,
      title_localized,
      abstract_localized,
      claims_localized,
      description_localized,
      publication_date
    ) ORDER BY publication_date DESC, kind_code DESC LIMIT 1)[OFFSET(0)] AS latest_pub
  FROM
    filtered_publications
  GROUP BY
    base_pub_number
)
--step 4: perform the left join
SELECT
  pl.patent_id,
  latest.kind_code,
  latest.family_id,
  latest.title_localized[SAFE_OFFSET(0)].text as title,
  latest.abstract_localized[SAFE_OFFSET(0)].text as abstract,
  latest.claims_localized[SAFE_OFFSET(0)].text as claims,
  latest.description_localized[SAFE_OFFSET(0)].text as description,
  latest.publication_date
FROM
  patent_list pl
LEFT JOIN
  latest_by_patent lbp
ON
  pl.patent_id = lbp.base_pub_number
CROSS JOIN
  UNNEST([lbp.latest_pub]) AS latest