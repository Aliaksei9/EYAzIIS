import time

class PerformanceTimer:
    """Простой статический класс для измерения времени выполнения кода"""
    
    _start_time = None
    
    @staticmethod
    def start():
        """Запустить таймер"""
        PerformanceTimer._start_time = time.time()
    
    @staticmethod
    def stop(message="Время выполнения"):
        if PerformanceTimer._start_time is None:
            print("Таймер не был запущен!")
            return None
        elapsed_time = time.time() - PerformanceTimer._start_time
        print(f"{message}: {elapsed_time:.4f} секунд")
        PerformanceTimer._start_time = None
        return elapsed_time   # <<< добавить возврат значения