SELECT SUM(x_min), COUNT(frame) FROM objects00 WHERE x_min > 1000;
SELECT AVG(x_min), COUNT(*) FROM objects00 WHERE x_min > 1000;
SELECT SUM(x_min), SUM(y_max), AVG(x_max), COUNT(*) FROM objects00 WHERE y_min > 500;
SELECT SUM(x_min), SUM(y_max), SUM(x_max), SUM(y_min) FROM objects00 WHERE x_min < 1000;
SELECT AVG(x_min), SUM(y_max), AVG(x_max), SUM(y_min) FROM objects00 WHERE frame > 100000;
SELECT COUNT(x_min), SUM(y_max), COUNT(x_max), AVG(y_min) FROM objects00 WHERE x_min > 700 AND y_min > 700;
SELECT SUM(x_min) FROM objects00 WHERE x_min > 1000;
SELECT COUNT(x_min) FROM objects00 WHERE x_min > 1000;
SELECT AVG(x_min) FROM objects00 WHERE x_min > 1000;
SELECT SUM(x_min) FROM objects00;