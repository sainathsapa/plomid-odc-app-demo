-- ODC migration V004: rename odc_seat_layout.version to layout_version.
-- Bare version collides with the PLOMID version() builtin: every write
-- stored the server version string and every read returned it. Qualified
-- layout_version behaves like a normal column. Existing rows are repaired
-- by copying layout_name (hand saves and seed imports both version by
-- name, so no information is lost).
ALTER TABLE odc_seat_layout RENAME COLUMN version TO layout_version;
UPDATE odc_seat_layout SET layout_version = layout_name;
