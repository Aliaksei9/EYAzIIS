from constants import POS_TRANSLATE, GRAMMEMES_TRANSLATE, SYNTAX_DEP_TRANSLATE, SYNTAX_POS_TRANSLATE


class MorphTagTranslator:
    POS_TRANSLATE = POS_TRANSLATE
    GRAMMEMES_TRANSLATE = GRAMMEMES_TRANSLATE

    def translate(self, tag_str):
        if not tag_str or tag_str == 'None':
            return ''
        parts = str(tag_str).replace(',', ', ').split()
        translated = []
        for part in parts:
            part = part.strip(', ')
            sub = [self.GRAMMEMES_TRANSLATE.get(p.strip(), "") for p in part.split(',')]
            translated.append(', '.join(sub))
        return ' '.join(translated).strip()

    def get_pos_rus(self, pos):
        return self.POS_TRANSLATE.get(pos, pos or "NONE")


class SyntaxTagTranslator:
    DEP_TRANSLATE = SYNTAX_DEP_TRANSLATE
    POS_TRANSLATE = SYNTAX_POS_TRANSLATE

    def get_dep_rus(self, dep):
        if not dep:
            return 'неизвестно'
        return self.DEP_TRANSLATE.get(dep, dep)

    def get_pos_rus(self, pos):
        if not pos:
            return 'неизвестно'
        return self.POS_TRANSLATE.get(pos, pos)

    def translate_token(self, token_data):
        return {
            'text': token_data.get('text', ''),
            'lemma': token_data.get('lemma', ''),
            'pos_rus': self.get_pos_rus(token_data.get('pos', '')),
            'tag': token_data.get('tag', ''),
            'dep_rus': self.get_dep_rus(token_data.get('dep', '')),
            'dep': token_data.get('dep', ''),
            'head': token_data.get('head', ''),
            'head_idx': token_data.get('head_idx', 0)
        }
