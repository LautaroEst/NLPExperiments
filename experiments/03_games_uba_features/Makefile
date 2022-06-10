
include config.mk

export PYTHONPATH:=../../:${PYTHONPATH}
SCRIPTS_DIR:=../../scripts
out_dir=output

# Configs files
TOKENIZER_CONFIG = configs/tokenizers/$(tokenizer).py
FEATURES_CONFIG = configs/features/$(features).py
DATA_CONFIG = configs/data/$(data).py

# Output directories
TOKENIZER_OUT_DIR = $(out_dir)/$(shell basename $(tokenizer))
FEATURES_OUT_DIR = $(out_dir)/$(shell basename $(tokenizer))/$(shell basename $(features))
DATA_OUT_DIR = $(out_dir)/$(shell basename $(tokenizer))/$(shell basename $(features))/$(shell basename $(data))
RESULTS_DIR = $(out_dir)/$(shell basename $(tokenizer))/$(shell basename $(features))/$(shell basename $(data))/results

# Directories for the output files
TOKENIZER_FILES_DIR = $(TOKENIZER_OUT_DIR)/files
FEATURES_FILES_DIR = $(FEATURES_OUT_DIR)/files
DATA_FILES_DIR = $(DATA_OUT_DIR)/files

# make all performs full experiment
all: tokenizer features data inspect

# Tasks aliases
tokenizer: $(TOKENIZER_FILES_DIR)
features: $(FEATURES_FILES_DIR)
data: $(DATA_FILES_DIR)
inspect: $()


$(TOKENIZER_FILES_DIR): $(TOKENIZER_CONFIG)
	@echo "Preparing tokenizer ${tokenizer}..."
	@rm -rf $(TOKENIZER_FILES_DIR)
	@mkdir -p $(TOKENIZER_FILES_DIR)
	@python $(SCRIPTS_DIR)/prepare_tokenizer.py \
		--config $(TOKENIZER_CONFIG) \
		--out $(TOKENIZER_FILES_DIR)


$(FEATURES_FILES_DIR): $(FEATURES_CONFIG) $(TOKENIZER_FILES_DIR)
	@echo "Preparing features ${features}..."
	@rm -rf $(FEATURES_FILES_DIR)
	@mkdir -p $(FEATURES_FILES_DIR)
	@python $(SCRIPTS_DIR)/prepare_features.py \
		--config $(FEATURES_CONFIG) \
		--tokenizer_dir $(TOKENIZER_FILES_DIR) \
		--out $(FEATURES_FILES_DIR)
	@echo ""


$(DATA_FILES_DIR): $(DATA_CONFIG) $(TOKENIZER_FILES_DIR)
	@echo "Preprocessing ${data} data..."
	@rm -rf $(DATA_FILES_DIR)
	@mkdir -p $(DATA_FILES_DIR)
	@python $(SCRIPTS_DIR)/prepare_supervised_sequences.py \
		--config $(DATA_CONFIG) \
		--tokenizer_dir $(TOKENIZER_FILES_DIR) \
		--out $(DATA_FILES_DIR)
	@echo ""


$(RESULTS_DIR): $(TOKENIZER_FILES_DIR) $(FEATURES_FILES_DIR) $(DATA_FILES_DIR)
	@echo "Starting inspection..."
	@rm -rf $(RESULTS_DIR)
	@mkdir -p $(RESULTS_DIR)
	@python $(SCRIPTS_DIR)/inspect_features.py \
		--tokenizer_dir $(TOKENIZER_FILES_DIR) \
		--features_dir $(FEATURES_FILES_DIR) \
		--data_dir $(DATA_FILES_DIR) \
		--out $(TRAINING_FILES_DIR)


clean: 
	rm -rf $(out_dir)