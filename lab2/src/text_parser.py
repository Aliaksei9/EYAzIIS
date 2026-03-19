import re
import pymorphy3
from razdel import tokenize
from tag_translator import TagTranslator


class TextParser:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer(lang='ru')
        self.translator = TagTranslator()

    def analyze_text(self, text: str, doc_id: int):
        tokens = [token.text for token in tokenize(text) if re.match(r'^\w+$', token.text)]
        rows = []
        for token in tokens:
            if token.isdigit():
                continue
            p = self.morph.parse(token)[0]
            pos = p.tag.POS or 'UNK'
            morph_tag = str(p.tag)
            rows.append({
                'doc_id': doc_id,
                'wordform': token.lower(),
                'lemma': p.normal_form.lower(),
                'pos_rus': self.translator.get_pos_rus(pos),
                'morph_rus': self.translator.translate(morph_tag),
            })
        return rows
