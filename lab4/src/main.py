import time
import matplotlib.pyplot as plt
from corpus_manager import CorpusManager
from performance_timer import PerformanceTimer


def count_tokens_in_text(text):
    """Подсчитывает количество токенов в тексте."""
    return len(text.split())


def measure_semantic_analysis_performance(manager, doc_ids):
    """
    Замеряет время выполнения семантического анализа для каждого документа.
    Возвращает список кортежей: (doc_title, token_count, cao_time, causal_time, total_time)
    """
    performance_data = []

    for doc_id in doc_ids:
        docs = manager.get_doc_list()
        doc_title = docs[doc_id]['metadata']['title'] if doc_id in docs else f"Doc {doc_id}"

        text = manager.storage.get_document_text(doc_id)
        if not text:
            continue

        token_count = count_tokens_in_text(text)

        # Замер extract_cao
        PerformanceTimer.start()
        cao_results = manager.semantic_analyzer.extract_cao(text)
        cao_time = time.time() - PerformanceTimer._start_time if PerformanceTimer._start_time else 0
        PerformanceTimer._start_time = None

        # Замер extract_causal_chains
        PerformanceTimer.start()
        causal_chains = manager.semantic_analyzer.extract_causal_chains(text)
        causal_time = time.time() - PerformanceTimer._start_time if PerformanceTimer._start_time else 0
        PerformanceTimer._start_time = None

        total_time = cao_time + causal_time

        performance_data.append((doc_title, token_count, cao_time, causal_time, total_time))
        print(f"{doc_title}: Tokens={token_count}, CAO={cao_time:.4f}s, Causal={causal_time:.4f}s, Total={total_time:.4f}s")

    return performance_data


def build_performance_chart(performance_data, save_path="semantic_analysis_performance.png"):
    """
    Строит график зависимости времени выполнения от количества токенов.
    """
    if not performance_data:
        print("Нет данных для построения графика")
        return

    # Подготовка данных
    token_counts = [data[1] for data in performance_data]
    cao_times = [data[2] for data in performance_data]
    causal_times = [data[3] for data in performance_data]
    total_times = [data[4] for data in performance_data]
    doc_titles = [data[0] for data in performance_data]

    # Сортировка по количеству токенов
    sorted_data = sorted(zip(token_counts, cao_times, causal_times, total_times, doc_titles))
    token_counts, cao_times, causal_times, total_times, doc_titles = zip(*sorted_data)

    # Создание графика
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(token_counts, cao_times, 'o-', label='Извлечение CAO', color='skyblue', linewidth=2, markersize=8)
    ax.plot(token_counts, causal_times, 's-', label='Причинно-следственные связи', color='lightcoral', linewidth=2, markersize=8)
    ax.plot(token_counts, total_times, '^-', label='Общее время', color='lightgreen', linewidth=2, markersize=8)

    ax.set_xlabel('Количество токенов')
    ax.set_ylabel('Время выполнения (секунды)')
    ax.set_title('Производительность семантического анализа\n(зависимость времени от количества токенов)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Подписи точек
    for i, (tokens, total, title) in enumerate(zip(token_counts, total_times, doc_titles)):
        short_title = title[:20] + "..." if len(title) > 20 else title
        ax.annotate(short_title, (tokens, total), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=7, alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nГрафик сохранён: {save_path}")
    return save_path


if __name__ == "__main__":
    manager = CorpusManager()

    # Запуск приложения
    manager.run()
