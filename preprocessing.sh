#!/bin/bash

codeprep basic --path data/BCB/dataset/ --output-path data/BCB/dataset_basic/ --ext "java" --no-str --no-com --no-spaces --no-unicode --no-case --calc-vocab
#codeprep bpe 10k --path data/BCB/dataset/ --ext "java" --no-str --no-com --no-spaces --no-unicode --calc-vocab
