import os
import re
import json
import logging
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

_model_cache = {}

class MusicChatBot:
    def __init__(self, data_storage):
        self.storage = data_storage
        self._setup_model()

    def _setup_model(self):
        global _model_cache
        if "model" not in _model_cache:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            print(f"Загрузка модели {self.model_name} на {self.device}...")
            _model_cache["tokenizer"] = AutoTokenizer.from_pretrained(self.model_name)
            _model_cache["model"] = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            _model_cache["device"] = self.device
            print(f"Модель загружена. GPU: {torch.cuda.get_device_name(0) if self.device == 'cuda' else 'Не используется'}")
            
        self.tokenizer = _model_cache["tokenizer"]
        self.model = _model_cache["model"]
        self.device = _model_cache["device"]

    def extract_search_keywords(self, query: str) -> str:
        system_prompt = (
            "Ты — поисковый оптимизатор. Извлеки из запроса пользователя ТОЛЬКО ключевые слова для полнотекстового поиска в базе данных. "
            "Убери всё лишнее: вежливые обращения, местоимения, вопросительные слова, вводные конструкции, просьбы. "
            "Верни результат СТРОГО в формате: ключевые слова через пробел. Никаких пояснений, знаков препинания или кавычек."
        )
        user_msg = f"Запрос: {query}\nКлючевые слова:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        cleaned = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        cleaned = re.sub(r'[^\w\sа-яА-ЯёЁ0-9]', '', cleaned)
        cleaned = " ".join(cleaned.split())

        print(f"ЗАПРОС ПОЛЬЗОВАТЕЛЯ: '{query}'\nОЧИЩЕННЫЙ ДЛЯ FTS: '{cleaned if cleaned else query}'\n")
        
        return cleaned if cleaned else query

    def _find_cluster_within_span(self, doc_positions: dict, max_span: int = 20):
        for doc_id, word_map in doc_positions.items():
            events = sorted((idx, w) for w, idxs in word_map.items() for idx in idxs)
            if not events:
                continue

            word_set = set(word_map.keys())
            
            clusters = []
            current_cluster = [events[0]]
            for i in range(1, len(events)):
                if events[i][0] - events[i-1][0] <= max_span:
                    current_cluster.append(events[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [events[i]]
            clusters.append(current_cluster)

            for cluster in clusters:
                found_words = {w for _, w in cluster}
                if found_words >= word_set:
                    cluster_indices = [idx for idx, _ in cluster]
                    return doc_id, cluster_indices
                    
        return None, []

    def _collect_context(self, matches: list, window: int = 10, max_total: int = 50) -> list:
        matches.sort(key=lambda x: x[1])
        seen, context = set(), []
        for doc_id, sent_idx in matches:
            for s in self.storage.get_sentence_context_window(doc_id, sent_idx, window):
                if s not in seen:
                    seen.add(s)
                    context.append(s)
            if len(context) >= max_total: break
        return context[:max_total]

    def _log_context(self, step: str, query: str, context: list):
        print("\n" + "="*60)
        print(f"ЗАПРОС FTS: {query}")
        steps = {
            "stage1_exact": "ЭТАП 1: Точное совпадение всех слов в одном предложении (топ-5)",
            "stage2_cluster": "ЭТАП 2: Кластерный поиск (слова в пределах 20 предложений)",
            "stage3_doc": "ЭТАП 3: Документный поиск (все слова в одном документе)"
        }
        print(steps.get(step, step))
        print(f"Передано предложений в LLM: {len(context)}")
        print("-" * 60)
        print("\n".join(context))
        print("="*60 + "\n")

    def retrieve_context(self, query: str, window: int = 10, max_total_sentences: int = 50) -> str:
        matches1 = self.storage.find_sentences_matching_all_words(query, top_k=5)
        if matches1:
            ctx = self._collect_context(matches1, 5, max_total_sentences)
            self._log_context("stage1_exact", query, ctx)
            return "\n".join(ctx)

        words = re.findall(r'[а-яА-ЯёЁa-zA-Z]{3,}', query)
        if len(words) < 2:
            return ""

        doc_positions = {}
        for w in words:
            for doc_id, sent_idx in self.storage.get_positions_for_word(w, max_results=30):
                doc_positions.setdefault(doc_id, {}).setdefault(w, []).append(sent_idx)

        if doc_positions:
            doc_id, cluster_indices = self._find_cluster_within_span(doc_positions, max_span=20)
            if doc_id:
                matches2 = [(doc_id, idx) for idx in cluster_indices]
                ctx = self._collect_context(matches2, window, max_total_sentences)
                if ctx:
                    self._log_context("stage2_cluster", query, ctx)
                    return "\n".join(ctx)

        common_docs = self.storage.get_docs_with_all_words(words, 2)
        if common_docs:
            matches3 = []
            for doc_id in common_docs:
                for w in words:
                    matches3.extend([(d, i) for d, i in self.storage.get_positions_for_word(w, max_results=20) if d == doc_id])
            if matches3:
                ctx = self._collect_context(matches3, window, max_total_sentences)
                self._log_context("stage3_doc", query, ctx)
                return "\n".join(ctx)

        return ""

    def generate_answer(self, context: str, query: str) -> str:
        system_prompt = (
            "Ты — музыкальный ассистент. Отвечай на вопросы пользователя, опираясь ТОЛЬКО на предоставленный контекст. "
            "Если в контексте нет ответа, вежливо сообщи об этом. Отвечай кратко, по делу и на русском языке."
        )
        messages = [
            {"role": "system", "content": system_prompt + f"\nКонтекст:\n{context}"},
            {"role": "user", "content": query}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response_text.strip()

    def get_response(self, user_input: str) -> str:
        search_query = self.extract_search_keywords(user_input)
        print(search_query)
        
        context = self.retrieve_context(search_query, 10, 50)
        
        if not context:
            return "К сожалению, в загруженной базе текстов не нашлось релевантной информации по вашему запросу. Попробуйте переформулировать вопрос или загрузите больше текстов про музыку."
            
        return self.generate_answer(context, user_input)