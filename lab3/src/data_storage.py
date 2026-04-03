import json
import pandas as pd
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import sessionmaker
import streamlit as st
from models import Base, CorpusDocument, Sentence, Token
from performance_timer import PerformanceTimer


class DataStorage:
    def __init__(self):
        self.engine = create_engine(
            st.secrets['database']['url'],
            connect_args={'client_encoding': 'utf8'}
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_sentences_and_tokens(self, sentence_rows, token_rows):
        if sentence_rows:
            with self.Session() as session:
                session.bulk_insert_mappings(Sentence, sentence_rows)
                session.commit()
                sentences = session.query(Sentence.id, Sentence.doc_id, Sentence.sentence_index).all()
                sentence_id_map = {(row.doc_id, row.sentence_index): row.id for row in sentences}
                for token in token_rows:
                    temp_sent_id = token['sentence_id']
                    doc_id = token['doc_id']
                    real_sentence_id = sentence_id_map.get((doc_id, temp_sent_id))
                    if real_sentence_id:
                        token['sentence_id'] = real_sentence_id
                session.bulk_insert_mappings(Token, token_rows)
                session.commit()

    def add_document(self, text, metadata):
        doc = CorpusDocument(
            title=metadata['title'],
            author=metadata['author'],
            year=metadata['year'],
            type=metadata['type'],
            text=text,
            filename=metadata.get('filename')
        )
        with self.Session() as session:
            session.add(doc)
            session.commit()
            session.refresh(doc)
            return doc.id

    def update_document_text(self, doc_id, new_text):
        with self.Session() as session:
            doc = session.query(CorpusDocument).filter(CorpusDocument.id == doc_id).first()
            if doc:
                doc.text = new_text
                session.commit()

    def get_document_metadata(self, doc_id):
        with self.Session() as session:
            doc = session.query(CorpusDocument).filter(CorpusDocument.id == doc_id).first()
            if doc:
                return {
                    'title': doc.title,
                    'author': doc.author,
                    'year': doc.year,
                    'type': doc.type
                }
            return None

    def get_document_text(self, doc_id):
        with self.Session() as session:
            doc = session.query(CorpusDocument).filter(CorpusDocument.id == doc_id).first()
            return doc.text if doc else None

    def get_sentences_for_doc(self, doc_id):
        with self.Session() as session:
            sentences = session.query(Sentence).filter(Sentence.doc_id == doc_id)\
                .order_by(Sentence.sentence_index).all()
            return [{'id': s.id, 'text': s.text, 'sentence_index': s.sentence_index} for s in sentences]

    def get_tokens_for_sentence(self, sentence_id):
        with self.Session() as session:
            tokens = session.query(Token).filter(Token.sentence_id == sentence_id)\
                .order_by(Token.token_index).all()
            return [{'wordform': t.wordform, 'lemma': t.lemma, 'pos_rus': t.pos_rus,
                     'morph_rus': t.morph_rus, 'token_index': t.token_index} for t in tokens]

    def remove_data_for_doc(self, doc_id):
        with self.Session() as session:
            session.query(Token).filter(Token.doc_id == doc_id).delete()
            session.commit()
            session.query(Sentence).filter(Sentence.doc_id == doc_id).delete()
            session.commit()

    def get_doc_list(self):
        query = select(CorpusDocument.id, CorpusDocument.title)
        with self.engine.connect() as conn:
            result = conn.execute(query).fetchall()
        return {row[0]: {'metadata': {'title': row[1]}} for row in result}

    def get_top_wordforms(self, limit=20):
        query = "SELECT wordform, COUNT(*) as frequency FROM tokens GROUP BY wordform ORDER BY frequency DESC LIMIT :limit"
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {'limit': limit}).fetchall()
        return pd.DataFrame(result, columns=['Словоформа', 'Частота'])

    def get_top_lemmas(self, limit=20):
        query = "SELECT lemma, COUNT(*) as frequency FROM tokens GROUP BY lemma ORDER BY frequency DESC LIMIT :limit"
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {'limit': limit}).fetchall()
        return pd.DataFrame(result, columns=['Лемма', 'Частота'])

    def get_pos_distribution(self):
        query = "SELECT pos_rus, COUNT(*) as count FROM tokens GROUP BY pos_rus ORDER BY count DESC"
        with self.engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()
        return pd.DataFrame(result, columns=['Часть речи', 'Количество'])

    def get_top_morph(self, limit=15):
        query = "SELECT morph_rus, COUNT(*) as frequency FROM tokens WHERE morph_rus != '' AND morph_rus IS NOT NULL GROUP BY morph_rus ORDER BY frequency DESC LIMIT :limit"
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {'limit': limit}).fetchall()
        return pd.DataFrame(result, columns=['Морфологические характеристики', 'Частота'])

    def get_doc_stats(self):
        query = """
        SELECT d.title as doc_title, COUNT(t.id) as count
        FROM documents d
        LEFT JOIN tokens t ON d.id = t.doc_id
        GROUP BY d.title
        ORDER BY count DESC
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()
        return pd.DataFrame(result, columns=['Название документа', 'Количество словоформ'])

    def search_words(self, query):
        PerformanceTimer.start()
        q = f"%{query.lower()}%"
        sql = """
        SELECT d.title as doc_title, t.wordform, t.lemma, t.pos_rus, t.morph_rus, d.author, d.year, d.type
        FROM tokens t
        JOIN documents d ON t.doc_id = d.id
        WHERE t.wordform ILIKE :q OR t.lemma ILIKE :q
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), {'q': q}).fetchall()
        PerformanceTimer.stop("Поиск слова:")
        return pd.DataFrame(result, columns=['Название документа', 'Словоформа', 'Лемма', 'Часть речи',
                                             'Морфологические характеристики', 'Автор', 'Год', 'Тип текста'])

    def search_phrases(self, phrase):
        sql = """
        SELECT s.id, s.doc_id, s.text, d.title
        FROM sentences s
        JOIN documents d ON s.doc_id = d.id
        WHERE s.text ILIKE :q
        ORDER BY s.doc_id, s.sentence_index
        """
        with self.engine.connect() as conn:
            results = conn.execute(text(sql), {'q': f"%{phrase}%"}).fetchall()
        return results

    def get_filtered_data(self, author=None, doc_type=None, year=None):
        where_clauses = []
        params = {}
        if author and author != "Все":
            where_clauses.append("d.author = :author")
            params['author'] = author
        if doc_type and doc_type != "Все":
            where_clauses.append("d.type = :type")
            params['type'] = doc_type
        if year and year != "Все":
            where_clauses.append("d.year = :year")
            params['year'] = year
        where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        sql = f"""
        SELECT d.title as doc_title, t.wordform, t.lemma, t.pos_rus, t.morph_rus, d.author, d.year, d.type
        FROM tokens t
        JOIN documents d ON t.doc_id = d.id
        {where}
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params).fetchall()
        return pd.DataFrame(result, columns=['Название документа', 'Словоформа', 'Лемма', 'Часть речи',
                                             'Морфологические характеристики', 'Автор', 'Год', 'Тип текста'])

    def get_authors(self):
        query = select(func.distinct(CorpusDocument.author))
        with self.engine.connect() as conn:
            result = conn.execute(query).fetchall()
        return sorted([row[0] for row in result if row[0]])

    def get_types(self):
        query = select(func.distinct(CorpusDocument.type))
        with self.engine.connect() as conn:
            result = conn.execute(query).fetchall()
        return sorted([row[0] for row in result if row[0]])

    def get_years(self):
        query = select(func.distinct(CorpusDocument.year))
        with self.engine.connect() as conn:
            result = conn.execute(query).fetchall()
        return sorted([row[0] for row in result if row[0]], reverse=True)

    def save_to_json(self):
        with self.Session() as session:
            docs = session.query(CorpusDocument).all()
            sentences = session.query(Sentence).all()
            tokens = session.query(Token).all()
        return json.dumps({
            'documents': [{'title': d.title, 'author': d.author, 'year': d.year,
                           'type': d.type, 'text': d.text, 'filename': d.filename}
                          for d in docs],
            'sentences': [{'doc_id': s.doc_id, 'sentence_index': s.sentence_index,
                           'text': s.text} for s in sentences],
            'tokens': [{'sentence_id': t.sentence_id, 'doc_id': t.doc_id,
                        'token_index': t.token_index, 'wordform': t.wordform,
                        'lemma': t.lemma, 'pos_rus': t.pos_rus, 'morph_rus': t.morph_rus}
                       for t in tokens]
        }, ensure_ascii=False, indent=2).encode('utf-8')

    def load_from_json(self, data):
        loaded = json.loads(data)
        with self.Session() as session:
            session.query(Token).delete()
            session.query(Sentence).delete()
            session.query(CorpusDocument).delete()
            session.commit()
            doc_id_map = {}
            for doc_dict in loaded['documents']:
                doc = CorpusDocument(**doc_dict)
                session.add(doc)
                session.flush()
                doc_id_map[doc.id] = doc.id
            session.commit()
            for sent_dict in loaded.get('sentences', []):
                sentence = Sentence(**sent_dict)
                session.add(sentence)
                session.flush()
            session.commit()
            for token_dict in loaded.get('tokens', []):
                token = Token(**token_dict)
                session.add(token)
            session.commit()
