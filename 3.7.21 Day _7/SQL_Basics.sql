SELECT title, release_year
FROM films
WHERE (release_year >= 1990 AND release_year < 2000)
AND (language = 'French' OR language = 'Spanish')
and gross > 2000000;

SELECT title, release_year
FROM films
WHERE release_year BETWEEN 1990 AND 2000
AND budget > 100000000
AND (language = 'Spanish' or language = 'French');

SELECT count(*)
FROM films
WHERE language is null;

SELECT birthdate, name
FROM people
ORDER BY birthdate;

SELECT title
FROM films
WHERE release_year = 2000 or release_year = 2012
ORDER BY release_year;

SELECT *
FROM films
WHERE release_year > 2015 or release_year < 2015
ORDER BY duration;

SELECT title, gross
FROM films
WHERE title like 'M%'
ORDER BY title;

SELECT release_year, AVG(budget) AS avg_budget, AVG(gross) AS avg_gross
FROM films
WHERE release_year > 1990
GROUP BY release_year
HAVING AVG(budget) > 60000000
ORDER BY avg(gross) DESC;
