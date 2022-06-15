
include config.mk

export PYTHONPATH:=../../:${PYTHONPATH}
SCRIPTS_DIR:=../../scripts
out_dir=output

# Configs files
TOKENIZER_CONFIG = configs/tokenizers/$(tokenizer).py
FEATURES_CONFIG = configs/features/$(features).py
MODEL_CONFIG = configs/models/$(model).py
DATA_CONFIG = configs/data/$(data).py
TRAINING_CONFIG = configs/training/$(training).py

# Output directories
TOKENIZER_OUT_DIR = $(out_dir)/$(shell basename $(tokenizer))
DATA_OUT_DIR = $(out_dir)/$(shell basename $(tokenizer))/$(shell basename $(data))
FEATURES_OUT_DIR = $(out_dir)/$(shell basename $(tokenizer))/$(shell basename $(data))/$(shell basename $(features))
MODEL_OUT_DIR = $(out_dir)/$(shell basename $(tokenizer))/$(shell basename $(data))/$(shell basename $(features))/$(shell basename $(model))/
TRAINING_OUT_DIR = $(out_dir)/$(shell basename $(tokenizer))/$(shell basename $(data))/$(shell basename $(features))/$(shell basename $(model))/$(shell basename $(training))

# Directories for the output files
TOKENIZER_FILES_DIR = $(TOKENIZER_OUT_DIR)/files
DATA_FILES_DIR = $(DATA_OUT_DIR)/files
FEATURES_FILES_DIR = $(FEATURES_OUT_DIR)/files
MODEL_FILES_DIR = $(MODEL_OUT_DIR)/files
TRAINING_FILES_DIR = $(TRAINING_OUT_DIR)

# make all performs full experiment
all: tokenizer data features model training

# Tasks aliases
tokenizer: $(TOKENIZER_FILES_DIR)
data: $(DATA_FILES_DIR)
features: $(FEATURES_FILES_DIR)
model: $(MODEL_FILES_DIR)
training: $(TRAINING_FILES_DIR)


$(TOKENIZER_FILES_DIR): $(TOKENIZER_CONFIG)
	@echo "Preparing tokenizer ${tokenizer}..."
	@rm -rf $(TOKENIZER_FILES_DIR)
	@mkdir -p $(TOKENIZER_FILES_DIR)
	@python $(SCRIPTS_DIR)/tokenizer/prepare_tokenizer.py \
		--config $(TOKENIZER_CONFIG) \
		--out $(TOKENIZER_FILES_DIR)


$(DATA_FILES_DIR): $(DATA_CONFIG) $(TOKENIZER_FILES_DIR)
	@echo "Preprocessing ${data} data..."
	@rm -rf $(DATA_FILES_DIR)
	@mkdir -p $(DATA_FILES_DIR)
	@python $(SCRIPTS_DIR)/data/prepare_supervised_sequences.py \
		--config $(DATA_CONFIG) \
		--tokenizer_dir $(TOKENIZER_FILES_DIR) \
		--out $(DATA_FILES_DIR)
	@echo ""


$(FEATURES_FILES_DIR): $(FEATURES_CONFIG) $(TOKENIZER_FILES_DIR)
	@echo "Preparing features ${features}..."
	@rm -rf $(FEATURES_FILES_DIR)
	@mkdir -p $(FEATURES_FILES_DIR)
	@python $(SCRIPTS_DIR)/features/prepare_features.py \
		--config $(FEATURES_CONFIG) \
		--tokenizer_dir $(TOKENIZER_FILES_DIR) \
		--out $(FEATURES_FILES_DIR)
	@echo ""


$(MODEL_FILES_DIR): $(MODEL_CONFIG)
	@echo "Initializing model ${model} parameters..."
	@rm -rf $(MODEL_FILES_DIR)
	@mkdir -p $(MODEL_FILES_DIR)
	@python $(SCRIPTS_DIR)/model/prepare_model.py \
		--config $(MODEL_CONFIG) \
		--out $(MODEL_FILES_DIR)
	@echo ""


$(TRAINING_FILES_DIR): $(TRAINING_CONFIG) $(TOKENIZER_FILES_DIR) $(FEATURES_FILES_DIR) $(MODEL_FILES_DIR) $(DATA_FILES_DIR)
	@echo "Starting training with ${training} config file..."
	@rm -rf $(TRAINING_FILES_DIR)
	@mkdir -p $(TRAINING_FILES_DIR)
	@python $(SCRIPTS_DIR)/training/train_sequence_estimator.py \
		--training_config $(TRAINING_CONFIG) \
		--tokenizer_dir $(TOKENIZER_FILES_DIR) \
		--features_dir $(FEATURES_FILES_DIR) \
		--model_dir $(MODEL_FILES_DIR) \
		--data_dir $(DATA_FILES_DIR) \
		--out $(TRAINING_FILES_DIR)


clean: 
	rm -rf $(out_dir)