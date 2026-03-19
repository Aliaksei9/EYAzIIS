from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class CorpusDocument(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    title = Column(String(255))
    author = Column(String(255))
    year = Column(Integer)
    type = Column(String(100))
    text = Column(Text)
    filename = Column(String(255))


class Token(Base):
    __tablename__ = 'tokens'
    id = Column(Integer, primary_key=True)
    doc_id = Column(Integer)
    wordform = Column(String(100))
    lemma = Column(String(100))
    pos_rus = Column(String(50))
    morph_rus = Column(Text)
