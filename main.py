import argparse
import logging
from datetime import datetime
from src.data_preparation import prepare_data
from src.model_pretraining import pretrain_model
from src.pseudo_labeling import generate_pseudo_labels
from src.representation_learning import train_model
from src.evaluation import evaluate_model

# Настройка логирования
logging.basicConfig(
    filename=f"logs/imbanid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Режим отладки (1 эпоха, 1% данных)")
    parser.add_argument("--skip_train", action="store_true", help="Пропустить обучение")
    args = parser.parse_args()

    try:
        logging.info("=== Запуск ImbaNID ===")

        # 1. Подготовка данных
        data = prepare_data(sample_ratio=0.01 if args.debug else 1.0)
        logging.info("Данные успешно загружены")

        # 2. Предобучение
        if not args.skip_train:
            model = pretrain_model(data, epochs=1 if args.debug else 10)
            logging.info("Предобучение завершено")

            # 3. Генерация псевдометок
            pseudo_labels = generate_pseudo_labels(model, data)
            logging.info(f"Сгенерировано {len(pseudo_labels)} псевдометок")

            # 4. Обучение представлений
            train_model(model, data, pseudo_labels)
            logging.info("Обучение завершено")

        # 5. Оценка
        evaluate_model(model if not args.skip_train else None, data)
        logging.info("Оценка завершена")

    except Exception as e:
        logging.error(f"Ошибка: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()