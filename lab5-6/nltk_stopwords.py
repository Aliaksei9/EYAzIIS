import nltk
from nltk.corpus import stopwords

# Скачиваем список стоп-слов, если он еще не скачан
nltk.download('stopwords')

# Получаем список стоп-слов для русского языка
russian_stopwords = stopwords.words('russian')

# Пример: вывод первых 10 слов
print(russian_stopwords)
