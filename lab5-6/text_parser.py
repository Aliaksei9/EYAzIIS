import re
import pymorphy3
from razdel import tokenize
from translators import MorphTagTranslator


class TextParser:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer(lang='ru')
        self.translator = MorphTagTranslator()

    def analyze_text(self, text: str, doc_id: int):
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_rows = []
        token_rows = []
        for sent_idx, sent_text in enumerate(sentences):
            sentence_rows.append({
                'doc_id': doc_id,
                'sentence_index': sent_idx,
                'text': sent_text
            })
            tokens = [token.text for token in tokenize(sent_text) if re.match(r'^\w+$', token.text)]
            for token_idx, token in enumerate(tokens):
                if token.isdigit():
                    continue
                p = self.morph.parse(token)[0]
                pos = p.tag.POS or 'UNK'
                morph_tag = str(p.tag)
                token_rows.append({
                    'sentence_id': sent_idx,
                    'doc_id': doc_id,
                    'token_index': token_idx,
                    'wordform': token.lower(),
                    'lemma': p.normal_form.lower(),
                    'pos_rus': self.translator.get_pos_rus(pos),
                    'morph_rus': self.translator.translate(morph_tag),
                })
        return sentence_rows, token_rows
