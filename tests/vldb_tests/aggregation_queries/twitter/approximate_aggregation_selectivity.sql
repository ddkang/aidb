SELECT SUM(entity_start) FROM entity00 ERROR_TARGET 5% CONFIDENCE 95%;
SELECT SUM(entity_start) FROM entity00 WHERE type = 'CARDINAL' ERROR_TARGET 5% CONFIDENCE 95%;
SELECT SUM(entity_start) FROM entity00 WHERE type = 'ORG' ERROR_TARGET 5% CONFIDENCE 95%;
SELECT SUM(entity_start) FROM entity00 WHERE type = 'PERSON' ERROR_TARGET 5% CONFIDENCE 95%;
SELECT SUM(entity_start) FROM entity00 WHERE type = 'GPE' ERROR_TARGET 5% CONFIDENCE 95%;
