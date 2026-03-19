from constants import POS_TRANSLATE, GRAMMEMES_TRANSLATE


class TagTranslator:
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
