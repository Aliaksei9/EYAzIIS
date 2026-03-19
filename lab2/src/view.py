import streamlit as st
from constants import MIN_YEAR, MAX_YEAR, DEFAULT_YEAR, TERMINOLOGY_TEXT, HELP_TEXT, MENU_ITEMS
from performance_timer import PerformanceTimer


class View:
    def __init__(self, corpus_manager):
        self.corpus_manager = corpus_manager
        st.set_page_config(page_title="Корпусный менеджер", layout="wide")
        st.title("Корпусный менеджер текстов")

    def display_menu(self):
        menu = st.sidebar.selectbox(
            "Меню",
            MENU_ITEMS
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
            author = st.text_input("Автор", "Неизвестен")
            title_base = st.text_input("Базовое название", "Документ")
        with col2:
            year = st.number_input("Год", MIN_YEAR, MAX_YEAR, DEFAULT_YEAR)
            doc_type = st.selectbox("Тип текста", ["Художественный", "Научный", "Публицистический", "Другой"])

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

        tab_stats, tab_word, tab_phrase, tab_filter = st.tabs([
            "Общая статистика корпуса",
            "Поиск словоформ и лемм",
            "Поиск по фразе / кускам текста",
            "Фильтры по метаданным"
        ])

        with tab_stats:
            self.display_stats()

        with tab_word:
            query = self.render_word_search()
            self.display_word_search_results(query)

        with tab_phrase:
            phrase = self.render_phrase_search()
            self.display_phrase_search_results(phrase)

        with tab_filter:
            self.render_filters()

    def display_stats(self):
        st.subheader("Частотные характеристики по всему корпусу")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Топ-20 словоформ**")
            wf = self.corpus_manager.get_top_wordforms()
            st.dataframe(wf, use_container_width=True)

            st.write("**Топ-20 лемм**")
            lem = self.corpus_manager.get_top_lemmas()
            st.dataframe(lem, use_container_width=True)

        with col2:
            st.write("**Распределение по частям речи**")
            pos_df = self.corpus_manager.get_pos_distribution()
            st.dataframe(pos_df, use_container_width=True)

        st.write("**Морфологические характеристики (топ-15)**")
        morph_df = self.corpus_manager.get_top_morph()
        st.dataframe(morph_df, use_container_width=True)

        st.write("**Статистика по документам**")
        doc_stat = self.corpus_manager.get_doc_stats()
        st.dataframe(doc_stat, use_container_width=True)

    def render_word_search(self):
        query = st.text_input("Введите слово или лемму для поиска")
        return query

    def display_word_search_results(self, query):
        if query:
            result = self.corpus_manager.search_words(query)
            st.dataframe(result, use_container_width=True)

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

        for doc_id, doc_title, doc_text in results:
            contexts = self.corpus_manager.get_phrase_context(doc_text, phrase, words_count=80)

            if contexts:
                doc_title_display = doc_title or f"Документ {doc_id}"
                st.subheader(f"{doc_title_display}")

                for i, ctx in enumerate(contexts[:10], 1):
                    match_counter += 1
                    with st.expander(f"Находка #{match_counter}"):
                        left_display = f"...{ctx['left']}" if ctx['left'] else ""
                        right_display = f"{ctx['right']}..." if ctx['right'] else ""
                        st.markdown(f"{left_display}**{ctx['phrase']}**{right_display}")

                if len(contexts) > 10:
                    st.info(f"Показано 10 из {len(contexts)} совпадений в этом документе")

                st.divider()

        if match_counter == 0:
            st.info("Фраза не найдена ни в одном документе корпуса.")
        PerformanceTimer.stop("Поиск фразы:")

    def render_filters(self):
        st.subheader("Фильтрация по метаданным")
        authors = ["Все"] + self.corpus_manager.get_authors()
        types_list = ["Все"] + self.corpus_manager.get_types()
        years = ["Все"] + self.corpus_manager.get_years()

        col1, col2, col3 = st.columns(3)
        with col1:
            sel_author = st.selectbox("Автор", authors, key="filter_author")
        with col2:
            sel_type = st.selectbox("Тип текста", types_list, key="filter_type")
        with col3:
            sel_year = st.selectbox("Год", years, key="filter_year")

        apply_button = st.button("Применить", key="apply_filter_button")

        if apply_button:
            filtered = self.corpus_manager.get_filtered_data(sel_author, sel_type, sel_year)
            st.dataframe(filtered, use_container_width=True)

    def render_save_load(self):
        st.header("Сохранение и загрузка корпуса")
        col1, col2 = st.columns(2)
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
