
include config.mk

export PYTHONPATH:=../../:${PYTHONPATH}
SCRIPTS_DIR:=../../scripts
out_dir=output

# Configs files
TOKENIZER_CONFIG = configs/tokenizers/$(tokenizer).py
FEATURES_CONFIG = configs/features/$(features).py
DATA_CONFIG = configs/data/$(data).py
MODEL_CONFIG = configs/models/$(model).py
TRAINING_CONFIG = configs/training/$(training).py

# Output directories
TOKENIZER_OUT_DIR = $(out_dir)/$(shell basename $(tokenizer))
FEATURES_OUT_DIR = $(out_dir)/$(shell basename $(tokenizer))/$(shell basename $(features))
DATA_OUT_DIR = $(out_dir)/$(shell basename $(tokenizer))/$(shell basename $(features))/$(shell basename $(data))
MODEL_OUT_DIR = $(out_dir)/$(shell basename $(tokenizer))/$(shell basename $(features))/$(shell basename $(data))/$(shell basename $(model))
TRAINING_OUT_DIR = $(out_dir)/$(shell basename $(tokenizer))/$(shell basename $(features))/$(shell basename $(data))/$(shell basename $(model))/$(shell basename $(training))

# Directories for the output files
TOKENIZER_FILES_DIR = $(TOKENIZER_OUT_DIR)/files
FEATURES_FILES_DIR = $(FEATURES_OUT_DIR)/files
DATA_FILES_DIR = $(DATA_OUT_DIR)/files
TRAINING_FILES_DIR = $(TRAINING_OUT_DIR)/files

# make all performs full experiment
all: tokenizer features data training

# Tasks aliases
tokenizer: $(TOKENIZER_FILES_DIR)
features: $(FEATURES_FILES_DIR)
data: $(DATA_FILES_DIR)
training: $(TRAINING_FILES_DIR)


# $(out_dir):
# 	mkdir -p $(TRAINING_OUT_DIR)


$(TOKENIZER_FILES_DIR): $(TOKENIZER_CONFIG) #$(out_dir)
	rm -rf $(TOKENIZER_FILES_DIR)
	mkdir -p $(TOKENIZER_FILES_DIR)
	python $(SCRIPTS_DIR)/prepare_tokenizer.py \
		--config $(TOKENIZER_CONFIG) \
		--out $(TOKENIZER_FILES_DIR)


$(FEATURES_FILES_DIR): $(FEATURES_CONFIG) $(TOKENIZER_FILES_DIR)
	rm -rf $(FEATURES_FILES_DIR)
	mkdir -p $(FEATURES_FILES_DIR)
	python $(SCRIPTS_DIR)/prepare_features.py \
		--config $(FEATURES_CONFIG) \
		--tokenizer_dir $(TOKENIZER_FILES_DIR) \
		--out $(FEATURES_FILES_DIR)


$(DATA_FILES_DIR): $(DATA_CONFIG) $(TOKENIZER_FILES_DIR) $(FEATURES_FILE_DIR)
	rm -rf $(DATA_FILES_DIR)
	mkdir -p $(DATA_FILES_DIR)
	python $(SCRIPTS_DIR)/prepare_data_for_sequence_classification.py \
		--config $(DATA_CONFIG) \
		--tokenizer_dir $(TOKENIZER_FILES_DIR) \
		--out $(DATA_FILES_DIR)


$(TRAINING_FILES_DIR): $(TRAINING_CONFIG) $(MODEL_CONFIG) $(TOKENIZER_FILES_DIR) $(FEATURES_CONFIG) $(DATA_FILES_DIR)
	rm -rf $(TRAINING_FILES_DIR)
	mkdir -p $(TRAINING_FILES_DIR)
	python $(SCRIPTS_DIR)/train_sequence_classifier.py \
		--training_config $(TRAINING_CONFIG) \
		--model_config $(MODEL_CONFIG) \
		--tokenizer_dir $(TOKENIZER_FILES_DIR) \
		--features_config $(FEATURES_CONFIG) \
		--data_dir $(DATA_FILES_DIR) \
		--out $(TRAINING_FILES_DIR)


clean: 
	rm -rf $(out_dir)