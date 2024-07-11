SELECT AVG(num_textboxes) FROM arxiv00 ERROR_TARGET 5% CONFIDENCE 95%;
SELECT COUNT(*) FROM arxiv00 ERROR_TARGET 5% CONFIDENCE 95%;
SELECT AVG(num_textboxes) FROM arxiv00 WHERE num_textboxes > 10 ERROR_TARGET 5% CONFIDENCE 95%;
SELECT SUM(num_textboxes) FROM arxiv00 ERROR_TARGET 5% CONFIDENCE 95%;
SELECT AVG(num_textboxes) FROM arxiv00 WHERE pdf_id > 10000 ERROR_TARGET 5% CONFIDENCE 95%;
SELECT AVG(score) FROM sentiment01 ERROR_TARGET 5% CONFIDENCE 95%;
SELECT AVG(score), COUNT(*) FROM sentiment01 WHERE label = 'POSITIVE' ERROR_TARGET 5% CONFIDENCE 95%;
SELECT AVG(score), COUNT(*) FROM sentiment01 WHERE label = 'NEGATIVE' ERROR_TARGET 5% CONFIDENCE 95%;
SELECT COUNT(*) FROM sentiment01 WHERE label = 'NEUTRAL' ERROR_TARGET 5% CONFIDENCE 95%;
SELECT SUM(score) FROM sentiment01 WHERE label = 'POSITIVE' ERROR_TARGET 5% CONFIDENCE 95%;