.PHONY: prepare train eval clean

prepare:
	@echo "🔧 Preprocessing data and generating features..."
	python -c 'from src.data.preprocess import build_dataset; build_dataset()'

train:
	@echo "🏋️ Training model on all users..."
	python main.py train

eval:
	@echo "📈 Evaluating model..."
	python main.py eval

clean:
	@echo "🧹 Cleaning processed and features data..."
	rm -rf data/processed/*.csv
	rm -f data/features/all_users.csv