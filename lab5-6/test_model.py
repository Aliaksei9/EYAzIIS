from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Проверка доступности CUDA:", torch.cuda.is_available())
print("Версия transformers:", __import__('transformers').__version__)

try:
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    print("Загрузка модели...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        device_map="auto",
        dtype=torch.float16
    )
    print("✅ УСПЕХ! Модель загружена на GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("❌ ОШИБКА:", e)