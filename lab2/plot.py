import matplotlib.pyplot as plt

# Данные: количество слов и время выполнения (в секундах)
data = [
    (0, 0.0076),
    (238, 0.0370),
    (4284, 0.3790),
    (21420, 1.7850),
    (42840, 3.5963),
]

# Разделяем на два списка для удобства
words, times = zip(*sorted(data))  # сортируем по количеству слов

# Создаём график
plt.figure(figsize=(10, 6))
plt.plot(words, times, marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=8)

# Подписи и заголовок на русском
plt.title('Зависимость времени обработки корпуса от количества слов', fontsize=14, fontweight='bold')
plt.xlabel('Количество слов', fontsize=12)
plt.ylabel('Время выполнения (секунды)', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')

# Форматируем ось X для лучшей читаемости больших чисел
plt.ticklabel_format(style='plain', axis='x')

# Добавляем подписи к точкам
for w, t in zip(words, times):
    plt.annotate(f'{t:.4f} с', 
                 (w, t), 
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center', 
                 fontsize=9)

# Автоматически подгоняем макет, чтобы подписи не обрезались
plt.tight_layout()

# Показываем график
plt.show()