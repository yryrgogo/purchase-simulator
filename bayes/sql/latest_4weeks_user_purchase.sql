WITH
  -- 今週の日曜日を取得
  this_sunday AS (
    SELECT
      DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL MOD(EXTRACT(DAYOFWEEK FROM CURRENT_DATE())+5, 7) DAY), WEEK)
  ),
  -- 直近28日間の日付を生成
  date_range AS (
    SELECT
      DATE_SUB(DATE_TRUNC((SELECT * FROM this_sunday), WEEK), INTERVAL 1 * week_diff DAY) AS date
    FROM
      UNNEST(GENERATE_ARRAY(0, 27)) AS week_diff
    ORDER BY
      date
  )

SELECT
  *
FROM
  date_range
;
