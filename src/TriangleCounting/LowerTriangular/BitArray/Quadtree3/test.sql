drop database GridCSR; create database GridCSR; use GridCSR;

CREATE TABLE IF NOT EXISTS grids (
    id INT NOT NULL AUTO_INCREMENT,
    row INT NOT NULL,
    col INT NOT NULL,
    depth INT NOT NULL,
    shard_row INT NOT NULL,
    shard_col INT NOT NULL,
    range_row_from BIGINT NOT NULL, 
    range_row_to BIGINT NOT NULL, 
    range_col_from BIGINT NOT NULL, 
    range_col_to BIGINT NOT NULL, 
    row_byte BIGINT NOT NULL,
    ptr_byte BIGINT NOT NULL,
    col_byte BIGINT NOT NULL,
    stem VARCHAR(1024),
    PRIMARY KEY (id),
    UNIQUE KEY (row, col, depth, shard_row, shard_col)
);

CREATE TABLE IF NOT EXISTS cache (
    device_id INT NOT NULL,
    grid_id INT NOT NULL,
    file_type TINYINT DEFAULT NULL,
    state ENUM('NOTEXIST', 'LOADING', 'EXIST', 'EVICTING') DEFAULT 'NOTEXIST',
    ref_count INT DEFAULT 0,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,
    addr BIGINT DEFAULT NULL,
    byte BIGINT DEFAULT NULL,
    FOREIGN KEY (grid_id)
        REFERENCES grids (id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    UNIQUE KEY (device_id, grid_id, file_type)
);

INSERT IGNORE INTO cache (device_id, grid_id, file_type)
    SELECT *
    FROM
        (
            (SELECT -1 AS device_id, g.id AS grid_id FROM grids AS g) temp0
            CROSS JOIN
            (SELECT 'ROW' AS file_type UNION ALL SELECT 'PTR' UNION ALL SELECT 'COL') temp1
        );