-- depends: 0001_initial
-- __transactional__ = false

-- Triggers to keep claims_fts synced with the claims table.
-- Required because claims_fts uses content=claims (external content FTS5).

CREATE TRIGGER IF NOT EXISTS claims_ai AFTER INSERT ON claims BEGIN
    INSERT INTO claims_fts(rowid, subject_entity_id, predicate, object_value, tags)
    VALUES (new.rowid, new.subject_entity_id, new.predicate, new.object_value, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS claims_ad AFTER DELETE ON claims BEGIN
    INSERT INTO claims_fts(claims_fts, rowid, subject_entity_id, predicate, object_value, tags)
    VALUES ('delete', old.rowid, old.subject_entity_id, old.predicate, old.object_value, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS claims_au AFTER UPDATE ON claims BEGIN
    INSERT INTO claims_fts(claims_fts, rowid, subject_entity_id, predicate, object_value, tags)
    VALUES ('delete', old.rowid, old.subject_entity_id, old.predicate, old.object_value, old.tags);
    INSERT INTO claims_fts(rowid, subject_entity_id, predicate, object_value, tags)
    VALUES (new.rowid, new.subject_entity_id, new.predicate, new.object_value, new.tags);
END;
