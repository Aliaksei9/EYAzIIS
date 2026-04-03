import streamlit as st
import pandas as pd
import json
from constants import TERMINOLOGY_TEXT, HELP_TEXT, MENU_ITEMS, MIN_YEAR, MAX_YEAR, DEFAULT_YEAR
from performance_timer import PerformanceTimer


class View:
    def __init__(self, corpus_manager):
        self.corpus_manager = corpus_manager
        st.set_page_config(page_title="Корпусный менеджер", layout="wide")
        st.title("Корпусный менеджер текстов")

    def display_menu(self):
        if "menu_selection" not in st.session_state:
            st.session_state.menu_selection = MENU_ITEMS[0]
        menu = st.sidebar.selectbox(
            "Меню",
            MENU_ITEMS,
            key="menu_selection"
        )
        self.corpus_manager.handle_menu_selection(menu)

    def render_upload(self):
        st.header("Загрузка текстов в корпус")
        uploaded_files = st.file_uploader(
            "Выберите файлы (TXT, PDF, DOCX, RTF)",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'rtf']
        )
        st.subheader("Метаданные")
        col1, col2 = st.columns(2)
        with col1:
            author = st.text_input("Автор", "Неизвестен", key="author_input")
            title_base = st.text_input("Базовое название", "Документ", key="title_input")
        with col2:
            year = st.number_input("Год", MIN_YEAR, MAX_YEAR, DEFAULT_YEAR, key="year_input")
            doc_type = st.selectbox("Тип текста", ["Художественный", "Научный", "Публицистический", "Другой"], key="doctype_input")
        process_button = st.button(" Обработать и добавить в корпус", type="primary", key="process_files_button")
        return uploaded_files, author, title_base, year, doc_type, process_button

    def render_view_edit(self):
        st.header("Просмотр и редактирование корпуса")
        docs = self.corpus_manager.get_doc_list()
        if not docs:
            st.info("Корпус пока пустой")
            return None, None, False
        selected_id = st.selectbox(
            "Выберите документ",
            list(docs.keys()),
            format_func=lambda x: f"{x} — {docs[x]['metadata']['title']}"
        )
        text = self.corpus_manager.get_doc_text(selected_id)
        edited_text = st.text_area("Редактировать текст", text, height=400)
        save_button = st.button("Сохранить изменения", key="save_changes_button")
        return selected_id, edited_text, save_button

    def render_search_analysis(self):
        st.header("Поиск и анализ корпуса")
        if not self.corpus_manager.has_documents():
            st.warning("Сначала загрузите документы в раздел «Загрузка»")
            return
        if "search_tab" not in st.session_state:
            st.session_state.search_tab = 0
        tab_names = [
            "Общая статистика корпуса",
            "Поиск словоформ и лемм",
            "Поиск по фразе / кускам текста",
            "Фильтры по метаданным"
        ]
        tabs = st.tabs(tab_names)
        with tabs[0]:
            self.display_stats()
        with tabs[1]:
            query = self.render_word_search()
            self.display_word_search_results(query)
        with tabs[2]:
            phrase = self.render_phrase_search()
            self.display_phrase_search_results(phrase)
        with tabs[3]:
            self.render_filters()

    def display_stats(self):
        st.subheader("Частотные характеристики по всему корпусу")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Топ-20 словоформ**")
            wf = self.corpus_manager.get_top_wordforms()
            st.dataframe(wf, width='stretch')
            st.write("**Топ-20 лемм**")
            lem = self.corpus_manager.get_top_lemmas()
            st.dataframe(lem, width='stretch')
        with col2:
            st.write("**Распределение по частям речи**")
            pos_df = self.corpus_manager.get_pos_distribution()
            st.dataframe(pos_df, width='stretch')
        st.write("**Морфологические характеристики (топ-15)**")
        morph_df = self.corpus_manager.get_top_morph()
        st.dataframe(morph_df, width='stretch')
        st.write("**Статистика по документам**")
        doc_stat = self.corpus_manager.get_doc_stats()
        st.dataframe(doc_stat, width='stretch')

    def render_word_search(self):
        query = st.text_input("Введите слово или лемму для поиска")
        return query

    def display_word_search_results(self, query):
        if query:
            result = self.corpus_manager.search_words(query)
            st.dataframe(result, width='stretch')

    def render_phrase_search(self):
        st.subheader("Поиск по произвольному тексту / словосочетанию")
        phrase = st.text_input("Введите фразу или кусок текста", placeholder="Пользовательский интерфейс")
        return phrase

    def display_phrase_search_results(self, phrase: str):
        PerformanceTimer.start()
        if not phrase:
            return
        st.write(f"**Результаты поиска:** `{phrase}`")
        results = self.corpus_manager.search_phrases(phrase)
        if not results:
            st.info("Фраза не найдена ни в одном документе корпуса.")
            return
        match_counter = 0
        current_doc = None
        for sent_id, doc_id, sent_text, doc_title in results:
            doc_title_display = doc_title or f"Документ {doc_id}"
            if current_doc != doc_id:
                st.subheader(f"{doc_title_display}")
                current_doc = doc_id
            match_counter += 1
            with st.expander(f"Находка #{match_counter}: {sent_text[:80]}..."):
                st.markdown(f"**{sent_text}**")
        if match_counter == 0:
            st.info("Фраза не найдена ни в одном документе корпуса.")
        PerformanceTimer.stop("Поиск фразы:")

    def render_filters(self):
        st.subheader("Фильтрация по метаданным")
        authors = ["Все"] + self.corpus_manager.get_authors()
        types_list = ["Все"] + self.corpus_manager.get_types()
        years = ["Все"] + self.corpus_manager.get_years()
        if "filter_author" not in st.session_state:
            st.session_state.filter_author = "Все"
        if "filter_type" not in st.session_state:
            st.session_state.filter_type = "Все"
        if "filter_year" not in st.session_state:
            st.session_state.filter_year = "Все"
        col1, col2, col3 = st.columns(3)
        with col1:
            sel_author = st.selectbox("Автор", authors, key="filter_author")
        with col2:
            sel_type = st.selectbox("Тип текста", types_list, key="filter_type")
        with col3:
            sel_year = st.selectbox("Год", years, key="filter_year")
        apply_button = st.button("Применить фильтры", type="primary", key="apply_filters_button")
        if apply_button:
            filtered = self.corpus_manager.get_filtered_data(
                st.session_state.filter_author,
                st.session_state.filter_type,
                st.session_state.filter_year
            )
            st.dataframe(filtered, width='stretch')

    def render_syntax_analysis(self):
        st.header("Синтаксический анализ текста")
        if not self.corpus_manager.has_documents():
            st.warning("Сначала загрузите документы в корпус.")
            st.subheader("Или введите текст для анализа:")
            input_text = st.text_area("Введите текст для синтаксического разбора", height=200)
            if st.button("Анализировать введённый текст"):
                if input_text.strip():
                    with st.spinner("Выполняется синтаксический анализ..."):
                        sentences = self.corpus_manager.analyze_text_syntax(input_text)
                        self.display_syntax_results(sentences)
            return
        docs = self.corpus_manager.get_doc_list()
        selected_id = st.selectbox(
            "Выберите документ для анализа",
            list(docs.keys()),
            format_func=lambda x: f"{x} — {docs[x]['metadata']['title']}"
        )
        analyze_button = st.button("Выполнить синтаксический анализ документа")
        if analyze_button and selected_id is not None:
            with st.spinner("Выполняется синтаксический анализ..."):
                sentences = self.corpus_manager.analyze_document_syntax(selected_id)
                self.display_syntax_results(sentences)

    def display_syntax_results(self, sentences):
        if sentences is None:
            st.error("Ошибка при анализе. Убедитесь, что установлена модель spaCy.")
            return
        if not sentences:
            st.info("Нет данных для отображения.")
            return
        st.success(f"Анализ завершён. Обработано предложений: {len(sentences)}")
        result_data = []
        for sent in sentences:
            result_data.append({
                'sentence_id': sent['sentence_id'],
                'text': sent['text'],
                'root': sent['root'],
                'tokens': sent['tokens']
            })
        json_str = json.dumps(result_data, ensure_ascii=False, indent=2)
        st.download_button(
            label="Экспортировать результаты в JSON",
            data=json_str.encode('utf-8'),
            file_name="syntax_analysis.json",
            mime="application/json"
        )
        for i, sent in enumerate(sentences):
            with st.expander(f"Предложение {i+1}: {sent['text'][:100]}...", expanded=False):
                st.markdown(f"**{sent['text']}**")
                tab1, tab2, tab3 = st.tabs(["Дерево зависимостей", "Дерево составляющих", "Таблица"])
                with tab1:
                    if sent.get('html'):
                        st.components.v1.html(sent['html'], height=300, scrolling=True)
                    else:
                        st.warning("Дерево зависимостей недоступно")
                with tab2:
                    constituency = sent.get('constituency_tree')
                    if constituency and constituency.get('image_base64'):
                        st.image(f"data:image/png;base64,{constituency['image_base64']}",
                                caption="Дерево грамматических составляющих")
                    else:
                        st.warning("Дерево составляющих недоступно. Убедитесь, что модель spaCy загружена.")
                with tab3:
                    df_tokens = pd.DataFrame(sent['tokens'])
                    df_display = pd.DataFrame({
                        'Словоформа': df_tokens['wordform'],
                        'Лемма': df_tokens['lemma'],
                        'Часть речи': df_tokens['pos'],
                        'Тег': df_tokens['tag'],
                        'Зависимость': df_tokens['dependency'],
                        'Главное слово': df_tokens['head_word']
                    })
                    st.dataframe(df_display, width='stretch')

    def render_save_load(self):
        st.header("Сохранение и загрузка корпуса")
        col1, col2 = st.columns(2)
        save_button = False
        uploaded_json = None
        with col1:
            save_button = st.button("Сохранить корпус как JSON", key="save_json_button")
        with col2:
            uploaded_json = st.file_uploader("Загрузить сохранённый корпус", type='json')
        return save_button, uploaded_json

    def display_saved_json(self, data):
        st.download_button(
            label="Скачать corpus.json",
            data=data,
            file_name="corpus.json",
            mime="application/json",
            key="download_corpus"
        )

    def render_help(self):
        st.header("Система помощи")
        st.markdown(HELP_TEXT)

    def render_terminology(self):
        st.header("Терминологическая справка")
        st.markdown(TERMINOLOGY_TEXT)

    def render_sidebar_caption(self):
        st.sidebar.caption("Корпусный менеджер v0.999")
