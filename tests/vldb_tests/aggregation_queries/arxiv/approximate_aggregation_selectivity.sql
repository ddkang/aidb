SELECT AVG(num_textboxes) FROM arxiv00 ERROR_TARGET 5% CONFIDENCE 95%;
SELECT AVG(num_textboxes) FROM arxiv00 WHERE num_textboxes >= 15 ERROR_TARGET 5% CONFIDENCE 95%;
SELECT AVG(num_textboxes) FROM arxiv00 WHERE num_textboxes >= 27 ERROR_TARGET 5% CONFIDENCE 95%;
SELECT AVG(num_textboxes) FROM arxiv00 WHERE num_textboxes >= 42 ERROR_TARGET 5% CONFIDENCE 95%;
SELECT AVG(num_textboxes) FROM arxiv00 WHERE num_textboxes >= 75 ERROR_TARGET 5% CONFIDENCE 95%;
