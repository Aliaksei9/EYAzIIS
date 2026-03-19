import re
import streamlit as st
from sqlalchemy import text
from performance_timer import PerformanceTimer
from data_storage import DataStorage
from file_handler import FileHandler
from text_parser import TextParser
from view import View


class CorpusManager:
    def __init__(self):
        self.storage = DataStorage()
        self.file_handler = FileHandler()
        self.parser = TextParser()
        self.view = View(self)

    def add_document(self, text: str, metadata: dict):
        doc_id = self.storage.add_document(text, metadata)
        rows = self.parser.analyze_text(text, doc_id)
        self.storage.add_tokens(rows)
        return doc_id

    def update_document(self, doc_id: int, new_text: str):
        metadata = self.storage.get_document_metadata(doc_id)
        if metadata:
            self.storage.remove_tokens_for_doc(doc_id)
            rows = self.parser.analyze_text(new_text, doc_id)
            self.storage.add_tokens(rows)
            self.storage.update_document_text(doc_id, new_text)

    def process_uploaded_files(self, uploaded_files, author, title_base, year, doc_type):
        PerformanceTimer.start()
        for file in uploaded_files:
            text = self.file_handler.extract_text(file)
            if text:
                metadata = {
                    "title": f"{title_base} — {file.name}",
                    "author": author,
                    "year": year,
                    "type": doc_type,
                    "filename": file.name
                }
                doc_id = self.add_document(text, metadata)
                st.success(f"{file.name} добавлен (ID {doc_id})")
        PerformanceTimer.stop()

    def handle_menu_selection(self, menu):
        if menu == "Загрузка и построение корпуса":
            self.handle_upload()
        elif menu == "Просмотр и редактирование корпуса":
            self.handle_view_edit()
        elif menu == "Поиск и анализ":
            self.handle_search_analysis()
        elif menu == "Сохранение / Загрузка":
            self.handle_save_load()
        elif menu == "Терминологическая справка":
            self.handle_terminology()
        else:
            self.handle_help()
        self.view.render_sidebar_caption()

    def handle_upload(self):
        uploaded_files, author, title_base, year, doc_type, process_button = self.view.render_upload()
        if uploaded_files and process_button:
            self.process_uploaded_files(uploaded_files, author, title_base, year, doc_type)

    def handle_view_edit(self):
        selected_id, edited_text, save_button = self.view.render_view_edit()
        if save_button and selected_id is not None:
            self.update_document(selected_id, edited_text)
            st.success("Изменения сохранены и корпус обновлён")

    def handle_search_analysis(self):
        self.view.render_search_analysis()

    def handle_save_load(self):
        save_button, uploaded_json = self.view.render_save_load()
        if save_button:
            data = self.save_corpus_to_json()
            self.view.display_saved_json(data)
        if uploaded_json:
            data = uploaded_json.getvalue().decode('utf-8')
            self.load_corpus_from_json(data)
            st.success("Корпус успешно загружен!")

    def handle_terminology(self):
        self.view.render_terminology()

    def handle_help(self):
        self.view.render_help()

    def run(self):
        self.view.display_menu()

    def get_doc_text(self, doc_id: str) -> str:
        return self.storage.get_document_text(doc_id)

    def has_documents(self) -> bool:
        try:
            with self.storage.engine.connect() as conn:
                count = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
                return count > 0
        except Exception as e:
            print(f"Ошибка при проверке наличия документов: {e}")
            return False

    def get_phrase_context(self, text: str, phrase: str, words_count: int = 80):
        if not phrase or not text:
            return []

        results = []
        text_lower = text.lower()
        phrase_lower = phrase.lower()
        start_pos = 0

        while True:
            pos = text_lower.find(phrase_lower, start_pos)
            if pos == -1:
                break

            before_text = text[:pos]
            after_text = text[pos + len(phrase):]

            words_before = re.findall(r'\w+', before_text, re.UNICODE)
            words_before = words_before[-words_count:] if len(words_before) > words_count else words_before

            words_after = re.findall(r'\w+', after_text, re.UNICODE)
            words_after = words_after[:words_count] if len(words_after) > words_count else words_after

            if words_before:
                first_word = words_before[0]
                left_start = before_text.rfind(first_word)
                left_context = text[left_start:pos]
            else:
                left_context = ""

            if words_after:
                last_word = words_after[-1]
                right_end = after_text.find(last_word) + len(last_word)
                right_context = after_text[:right_end]
            else:
                right_context = ""

            context = {
                'left': left_context.strip(),
                'phrase': text[pos:pos + len(phrase)],
                'right': right_context.strip()
            }
            results.append(context)

            start_pos = pos + 1

        return results

    def get_doc_list(self):
        return self.storage.get_doc_list()

    def get_top_wordforms(self, limit=20):
        return self.storage.get_top_wordforms(limit)

    def get_top_lemmas(self, limit=20):
        return self.storage.get_top_lemmas(limit)

    def get_pos_distribution(self):
        return self.storage.get_pos_distribution()

    def get_top_morph(self, limit=15):
        return self.storage.get_top_morph(limit)

    def get_doc_stats(self):
        return self.storage.get_doc_stats()

    def search_words(self, query):
        return self.storage.search_words(query)

    def search_phrases(self, phrase):
        return self.storage.search_phrases(phrase)

    def get_filtered_data(self, author=None, doc_type=None, year=None):
        return self.storage.get_filtered_data(author, doc_type, year)

    def get_authors(self):
        return self.storage.get_authors()

    def get_types(self):
        return self.storage.get_types()

    def get_years(self):
        return self.storage.get_years()

    def save_corpus_to_json(self):
        return self.storage.save_to_json()

    def load_corpus_from_json(self, data):
        self.storage.load_from_json(data)
