import json
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, select, func, text
from sqlalchemy.orm import sessionmaker
from models import Base, CorpusDocument, Token
from performance_timer import PerformanceTimer


class DataStorage:
    def __init__(self):
        self.engine = create_engine(
            st.secrets['database']['url'],
            connect_args={'client_encoding': 'utf8'}
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_tokens(self, rows):
        if rows:
            with self.Session() as session:
                session.bulk_insert_mappings(Token, rows)
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

    def remove_tokens_for_doc(self, doc_id):
        with self.Session() as session:
            session.query(Token).filter(Token.doc_id == doc_id).delete()
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
        """Find documents containing the phrase using SQL ILIKE search."""
        sql = "SELECT id, title, text FROM documents WHERE text ILIKE :q"
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
            tokens = session.query(Token).all()

        return json.dumps({
            'documents': [{'id': d.id, 'title': d.title, 'author': d.author, 'year': d.year,
                        'type': d.type, 'text': d.text, 'filename': d.filename}
                        for d in docs],
            'tokens': [{'doc_id': t.doc_id, 'wordform': t.wordform, 'lemma': t.lemma,
                        'pos_rus': t.pos_rus, 'morph_rus': t.morph_rus}
                    for t in tokens]
        }, ensure_ascii=False, indent=2).encode('utf-8')

    def load_from_json(self, data):
        loaded = json.loads(data)

        with self.Session() as session:
            session.query(Token).delete()
            session.query(CorpusDocument).delete()
            session.commit()

            doc_id_map = {}
            for doc_dict in loaded['documents']:
                old_id = doc_dict.get('id')
                doc = CorpusDocument(
                    title=doc_dict['title'],
                    author=doc_dict['author'],
                    year=doc_dict['year'],
                    type=doc_dict['type'],
                    text=doc_dict['text'],
                    filename=doc_dict.get('filename')
                )
                session.add(doc)
                session.flush()
                doc_id_map[old_id] = doc.id

            for token_dict in loaded.get('tokens', []):
                old_doc_id = token_dict.get('doc_id')
                new_doc_id = doc_id_map.get(old_doc_id, old_doc_id)
                token = Token(
                    doc_id=new_doc_id,
                    wordform=token_dict['wordform'],
                    lemma=token_dict['lemma'],
                    pos_rus=token_dict['pos_rus'],
                    morph_rus=token_dict['morph_rus']
                )
                session.add(token)

            session.commit()
