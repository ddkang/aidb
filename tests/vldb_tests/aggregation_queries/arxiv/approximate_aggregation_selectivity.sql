SELECT AVG(num_textboxes) FROM arxiv00 ERROR_TARGET 5% CONFIDENCE 95%;
SELECT AVG(num_textboxes) FROM arxiv00 WHERE num_textboxes >= 16 ERROR_TARGET 5% CONFIDENCE 95%;
SELECT AVG(num_textboxes) FROM arxiv00 WHERE num_textboxes >= 28 ERROR_TARGET 5% CONFIDENCE 95%;
SELECT AVG(num_textboxes) FROM arxiv00 WHERE num_textboxes >= 45 ERROR_TARGET 5% CONFIDENCE 95%;
SELECT AVG(num_textboxes) FROM arxiv00 WHERE num_textboxes >= 79 ERROR_TARGET 5% CONFIDENCE 95%;
