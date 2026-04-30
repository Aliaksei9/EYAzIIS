import re
import spacy
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from constants import NON_PREDICATE_LEMMAS, CAUSE_MARKERS_WORDS, RESULT_MARKERS_WORDS

try:
    from wiki_ru_wordnet import WikiWordnet
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False


class SemanticAnalyzer:

    def __init__(self, model_name: str = "ru_core_news_sm"):
        self.model_name = model_name
        self.nlp = None
        self._load_spacy_model()

        self.wn = WikiWordnet() if WORDNET_AVAILABLE else None
        self._relations_cache = {}

        self.cause_markers_words = CAUSE_MARKERS_WORDS

        self.result_markers_words = RESULT_MARKERS_WORDS

    def _load_spacy_model(self):
        try:
            self.nlp = spacy.load(self.model_name)
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download", self.model_name])
            self.nlp = spacy.load(self.model_name)

        # Регистрируем расширения для кореференции
        if not spacy.tokens.Span.has_extension("coref"):
            spacy.tokens.Span.set_extension("coref", default=None)
        if not spacy.tokens.Token.has_extension("coref_cluster"):
            spacy.tokens.Token.set_extension("coref_cluster", default=None)

    def _apply_coref_resolution(self, doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
        sents = list(doc.sents)
        for i, sent in enumerate(sents):
            for tok in sent:
                if tok.pos_ == "PRON" and tok._.coref_cluster is None:
                    pron_gender = set(tok.morph.get("Gender", []))
                    pron_number = set(tok.morph.get("Number", []))
                    if not pron_gender or not pron_number:
                        continue

                    for prev_sent in reversed(sents[:i]):
                        for cand in prev_sent:
                            if cand.pos_ in ("NOUN", "PROPN"):
                                cand_gender = set(cand.morph.get("Gender", []))
                                cand_number = set(cand.morph.get("Number", []))
                                if pron_gender.issubset(cand_gender) and pron_number.issubset(cand_number):
                                    if tok.has_vector and cand.has_vector:
                                        sim = float(np.dot(tok.vector, cand.vector)) / \
                                              (np.linalg.norm(tok.vector) * np.linalg.norm(cand.vector) + 1e-8)
                                        if sim < 0.35:
                                            continue

                                    tok._.coref_cluster = cand.i
                                    span = doc[cand.i:cand.i + 1]
                                    span._.coref = {"antecedent": cand.text, "cluster_id": cand.i}
                                    break
                        if tok._.coref_cluster is not None:
                            break
        return doc


    def _get_subtree_text(self, token: spacy.tokens.Token) -> str:
        return " ".join(t.text for t in token.subtree if not t.is_punct)

    def _get_span_vector(self, token: spacy.tokens.Token) -> Optional[np.ndarray]:
        tokens = [t for t in token.subtree if not t.is_punct and t.has_vector]
        if not tokens:
            return token.vector if token.has_vector else None
        return sum(t.vector for t in tokens) / len(tokens)

    def _get_lexical_relations(self, lemma: str) -> Dict[str, List[str]]:
        if lemma in self._relations_cache:
            return self._relations_cache[lemma]
        syns, hyps = set(), set()
        if self.wn:
            try:
                for synset in self.wn.get_synsets(lemma):
                    for w in synset.get_words():
                        w_lemma = w.lemma().lower()
                        if w_lemma != lemma.lower():
                            syns.add(w_lemma)
                    for hyper in self.wn.get_hypernyms(synset):
                        for w in hyper.get_words():
                            hyps.add(w.lemma().lower())
            except Exception:
                pass
        result = {"synonyms": list(syns), "hypernyms": list(hyps)}
        self._relations_cache[lemma] = result
        return result

    def _find_predicates(self, tokens: List[spacy.tokens.Token]) -> List[spacy.tokens.Token]:
        candidates = []
        for t in tokens:
            if t.lemma_.lower() in NON_PREDICATE_LEMMAS:
                continue
            if t.pos_ in ("VERB", "AUX", "ADJ", "ADV"):
                candidates.append(t)
        candidates.sort(key=lambda t: (0 if t.pos_ in ("VERB", "AUX") else 1, t.i))
        return candidates

    def _resolve_pronoun_reference(self, pronoun: spacy.tokens.Token,
                                   context_sentences: List[spacy.tokens.Span]) -> Optional[str]:
        if pronoun.pos_ != "PRON":
            return None

        if pronoun._.coref_cluster is not None:
            return pronoun.doc[pronoun._.coref_cluster].text

        pron_gender = set(pronoun.morph.get("Gender", []))
        pron_number = set(pronoun.morph.get("Number", []))
        if not pron_gender or not pron_number:
            return None

        for sent in reversed(context_sentences[:-1]):
            for tok in sent:
                if tok.pos_ in ("NOUN", "PROPN"):
                    tok_gender = set(tok.morph.get("Gender", []))
                    tok_number = set(tok.morph.get("Number", []))
                    if pron_gender.issubset(tok_gender) and pron_number.issubset(tok_number):
                        return self._get_subtree_text(tok)
        return None

    def _extract_cao_for_verb(self, verb: spacy.tokens.Token, sent: spacy.tokens.Span,
                              require_subj_obj: bool = True) -> Optional[Dict[str, Any]]:
        if verb.lemma_.lower() in NON_PREDICATE_LEMMAS:
            return None

        subj, obj, iobj, prep_attr, adj_attr, adv_attr = "", "", "", "", "", ""
        passive_agent = ""

        for child in verb.children:
            dep = child.dep_
            if dep in ("nsubj", "nsubj:pass", "csubj"):
                subj = self._get_subtree_text(child)
            elif dep == "obj":
                obj = self._get_subtree_text(child)
            elif dep == "iobj":
                iobj = self._get_subtree_text(child)
            elif dep == "obl":
                case = next((c for c in child.children if c.dep_ == "case"), None)
                if case:
                    prep_attr = f"{case.text} {self._get_subtree_text(child)}"
                else:
                    iobj = self._get_subtree_text(child)
            elif dep == "obl:agent":
                #obl → обстоятельственный член
                passive_agent = self._get_subtree_text(child)
            elif dep == "advmod":
                #**advmod** (adverbial modifier) — обстоятельство, наречное определение
                adv_attr = self._get_subtree_text(child)
            #amod - адъективное определение, согласованное определение
            elif dep == "amod":
                head = child.head
                if head.dep_ in ("nsubj", "nsubj:pass", "obj", "iobj") or \
                        (head.dep_ == "ROOT" and head.pos_ in ("NOUN", "PROPN")):
                    adj_attr = self._get_subtree_text(child)
            elif dep == "nmod" and not adj_attr:
                # Несогласованное определение (родительный падеж, приложение)
                if child.head.dep_ in ("nsubj", "nsubj:pass", "obj", "iobj"):
                    adj_attr = self._get_subtree_text(child)

        if not subj and verb.dep_ == "advcl" and verb.head.pos_ == "VERB":
            for c in verb.head.children:
                if c.dep_ in ("nsubj", "nsubj:pass", "csubj"):
                    subj = self._get_subtree_text(c)
                    break

        if verb.dep_ == "ROOT" and "Pass" in verb.morph.get("Voice", []):
            if subj and passive_agent:
                obj = subj
                subj = passive_agent

        if verb.pos_ in ("ADJ", "ADV", "AUX"):
            require_subj_obj = False

        if verb.pos_ == "ADV" and not (subj or obj or iobj or prep_attr or adv_attr):
            if not any(child.dep_ == "fixed" for child in verb.children):
                return None

        if require_subj_obj and not (subj or obj):
            return None

        return {
            "subject": subj,
            "action": verb.text,
            "action_lemma": verb.lemma_,
            "object": obj,
            "attributes": {
                "preposition": prep_attr,
                "indirect_object": iobj,
                "adjective": adj_attr,
                "adverb": adv_attr,
                "passive_agent": passive_agent
            },
            "paradigmatic_relations": self._get_lexical_relations(verb.lemma_)
        }

    def extract_cao(self, text_or_doc: Union[str, spacy.tokens.Doc]) -> List[Dict[str, Any]]:
        if isinstance(text_or_doc, str):
            doc = self.nlp(text_or_doc)
        else:
            doc = text_or_doc
        results = []
        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "VERB":
                    cao = self._extract_cao_for_verb(token, sent)
                    if cao:
                        cao["sentence"] = sent.text
                        results.append(cao)
        return results

    def _find_causal_marker_in_sent(self, tokens: List[spacy.tokens.Token]) -> Optional[Tuple[str, str, int, int]]:
        n = len(tokens)
        for size in range(3, 0, -1):
            for i in range(n - size + 1):
                phrase = " ".join(t.text.lower() for t in tokens[i:i + size] if not t.is_punct)
                if phrase in self.cause_markers_words:
                    return ("cause", phrase, i, i + size - 1)
                if phrase in self.result_markers_words:
                    return ("result", phrase, i, i + size - 1)
        return None

    def _classify_single_token(self, token: spacy.tokens.Token) -> Optional[str]:
        text = token.text.lower()
        if text in self.cause_markers_words:
            return "cause"
        if text in self.result_markers_words:
            return "result"
        return None

    def _extract_gerund_cause(self, sent: spacy.tokens.Span) -> Optional[Dict[str, Any]]:
        tokens = list(sent)
        for i, tok in enumerate(tokens[:3]):
            if tok.pos_ == "VERB" and "VerbForm=Ger" in tok.tag_:
                if any(c.text.lower() == "не" and c.dep_ == "advmod" for c in tok.children):
                    continue
                later_verbs = [t for t in tokens if t.pos_ == "VERB" and t.i > tok.i]
                if later_verbs:
                    cause_cao = self._extract_cao_for_verb(tok, sent, require_subj_obj=False)
                    effect_cao = self._extract_cao_for_verb(later_verbs[0], sent, require_subj_obj=False)
                    if cause_cao and effect_cao:
                        return {
                            "type": "causal_chain", "cause_cao": cause_cao, "effect_cao": effect_cao,
                            "connector": tok.text, "direction": "cause→effect",
                            "sentence": sent.text, "scope": "intra-sentence", "method": "gerund"
                        }
                break
        return None

    def _are_semantically_related(self, verb1: spacy.tokens.Token, verb2: spacy.tokens.Token,
                                  threshold: float = 0.2) -> bool:
        if not verb1.has_vector or not verb2.has_vector:
            return True
        vec1 = self._get_span_vector(verb1)
        vec2 = self._get_span_vector(verb2)
        if vec1 is None or vec2 is None:
            return True
        sim = float(np.dot(vec1, vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
        return sim > threshold

    def _is_valid_cao(self, cao: Dict) -> bool:
        if not cao:
            return False
        if cao.get("action") or cao.get("subject") or cao.get("object"):
            return True
        attrs = cao.get("attributes", {})
        if attrs.get("indirect_object") or attrs.get("preposition"):
            return True
        return False

    def _extract_np_cause_from_marker(self, marker_end_token: spacy.tokens.Token,
                                      sent: spacy.tokens.Span) -> Optional[str]:
        for child in marker_end_token.children:
            if child.dep_ in ("obl", "nmod") and child.pos_ in ("NOUN", "PROPN"):
                return self._get_subtree_text(child)

        rest_tokens = [t.text for t in sent if t.i > marker_end_token.i and not t.is_punct and t.pos_ != 'AUX']
        return " ".join(rest_tokens) if rest_tokens else None

    def _extract_causal_chains(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        doc = self._apply_coref_resolution(doc)

        chains = []
        sentences = list(doc.sents)

        sent_predicates = []
        for sent in sentences:
            preds = self._find_predicates(sent)
            sent_predicates.append(preds)

        for sent_idx, sent in enumerate(sentences):
            sent_text = sent.text
            tokens = list(sent)
            preds = sent_predicates[sent_idx]

            marker_info = self._find_causal_marker_in_sent(tokens)
            if marker_info:
                marker_type, connector, start_idx, end_idx = marker_info

                global_start = tokens[start_idx].i
                global_end = tokens[end_idx].i
                marker_tokens = tokens[start_idx:end_idx + 1]

                # Определяем, является ли маркер предложным
                is_prepositional = (
                        any(t.pos_ == "ADP" for t in marker_tokens) and
                        not any(t.lemma_.lower() in {"что", "чтобы"} for t in marker_tokens)
                )

                # Фильтрация предикатов по глобальным индексам
                preds = [p for p in preds if not (global_start <= p.i <= global_end)]
                verbs_before = [v for v in preds if v.i < global_start]
                verbs_after = [v for v in preds if v.i > global_end]

                # Обработка предложных маркеров (именная причина)
                if is_prepositional:
                    cause_text = self._extract_np_cause_from_marker(tokens[end_idx], sent)
                    if cause_text and verbs_before:
                        effect_verb = verbs_before[0]
                        effect_cao = self._extract_cao_for_verb(effect_verb, sent, require_subj_obj=False)

                        if effect_cao:
                            cause_cao = {
                                "subject": "",
                                "action": "",
                                "action_lemma": "",
                                "object": cause_text,
                                "attributes": {},
                                "paradigmatic_relations": {}
                            }
                            chains.append({
                                "type": "causal_chain",
                                "cause_cao": cause_cao,
                                "effect_cao": effect_cao,
                                "connector": connector,
                                "direction": "cause→effect",
                                "sentence": sent_text,
                                "scope": "intra-sentence",
                                "method": "prepositional_cause"
                            })
                            continue

                # Стандартный случай с двумя глаголами
                if not verbs_before or not verbs_after:
                    continue

                if marker_type == "cause":
                    cause_candidates = verbs_after
                    effect_candidates = verbs_before
                else:  # result
                    cause_candidates = verbs_before
                    effect_candidates = verbs_after

                if cause_candidates and effect_candidates:
                    cause_verb = cause_candidates[0]
                    effect_verb = effect_candidates[0]

                    cause_cao = self._extract_cao_for_verb(cause_verb, sent, require_subj_obj=False)
                    effect_cao = self._extract_cao_for_verb(effect_verb, sent, require_subj_obj=False)

                    if (cause_cao and effect_cao and
                            self._is_valid_cao(cause_cao) and self._is_valid_cao(effect_cao) and
                            self._are_semantically_related(cause_verb, effect_verb)):
                        chains.append({
                            "type": "causal_chain",
                            "cause_cao": cause_cao,
                            "effect_cao": effect_cao,
                            "connector": connector,
                            "direction": "cause→effect",
                            "sentence": sent_text,
                            "scope": "intra-sentence",
                            "method": "improved_spacy"
                        })

            # Деепричастные обороты
            if not chains or chains[-1]["sentence"] != sent_text:
                gerund_cause = self._extract_gerund_cause(sent)
                if gerund_cause:
                    chains.append(gerund_cause)

        for i in range(len(sentences) - 1):
            prev_sent = sentences[i]
            curr_sent = sentences[i + 1]
            curr_tokens = list(curr_sent)

            first_token = curr_tokens[0] if curr_tokens else None
            if not first_token:
                continue

            marker_type = self._classify_single_token(first_token)
            connector = first_token.text

            if not marker_type and len(curr_tokens) > 1:
                bigram = f"{curr_tokens[0].text.lower()} {curr_tokens[1].text.lower()}"
                if bigram in self.result_markers_words:
                    marker_type = "result"
                    connector = bigram

            if marker_type == "result":
                prev_preds = sent_predicates[i]
                curr_preds = sent_predicates[i + 1]

                if prev_preds and curr_preds:
                    cause_verb = prev_preds[0]
                    effect_verb = curr_preds[0]

                    cause_cao = self._extract_cao_for_verb(cause_verb, prev_sent, require_subj_obj=False)
                    effect_cao = self._extract_cao_for_verb(effect_verb, curr_sent, require_subj_obj=False)

                    if cause_cao and effect_cao:
                        effect_subject = effect_cao.get("subject", "").strip().lower()
                        pronoun_token = None

                        if effect_subject in {"он", "она", "оно", "они", "её", "его", "их", "себя"}:
                            for tok in curr_sent:
                                if tok.pos_ == "PRON" and tok.text.lower() == effect_subject:
                                    pronoun_token = tok
                                    break

                        if pronoun_token:
                            ant = self._resolve_pronoun_reference(pronoun_token, sentences[:i + 2])
                            if ant:
                                effect_cao["subject"] = ant

                        if (self._is_valid_cao(cause_cao) and self._is_valid_cao(effect_cao) and
                                self._are_semantically_related(cause_verb, effect_verb)):
                            chains.append({
                                "type": "causal_chain",
                                "cause_cao": cause_cao,
                                "effect_cao": effect_cao,
                                "connector": connector,
                                "direction": "cause→effect",
                                "sentence": f"{prev_sent.text} {curr_sent.text}",
                                "scope": "inter-sentence",
                                "method": "improved_spacy"
                            })

        return chains

    def extract_causal_chains(self, text_or_doc: Union[str, spacy.tokens.Doc]) -> List[Dict[str, Any]]:
        if isinstance(text_or_doc, str):
            doc = self.nlp(text_or_doc)
        else:
            doc = text_or_doc
        return self._extract_causal_chains(doc)

    def analyze(self, text: str) -> Dict[str, Any]:
        doc = self.nlp(text)
        return {
            'cao_triads': self.extract_cao(doc),
            'causal_chains': self.extract_causal_chains(doc),
            'text': text
        }