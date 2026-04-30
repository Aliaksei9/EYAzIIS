ALTER TABLE sentences ADD COLUMN IF NOT EXISTS search_vector tsvector;

UPDATE sentences 
SET search_vector = to_tsvector('russian', text) 
WHERE search_vector IS NULL;

CREATE INDEX IF NOT EXISTS idx_sentences_search ON sentences USING GIN(search_vector);

CREATE TRIGGER tsvector_update 
BEFORE INSERT OR UPDATE OF text ON sentences
FOR EACH ROW EXECUTE FUNCTION tsvector_update_trigger(search_vector, 'pg_catalog.russian', text);