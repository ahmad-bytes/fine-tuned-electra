# Investigation and Mitigation of Gender Bias in the SNLI Dataset for Natural Language Inference

## Overview

This paper investigates gender bias in the SNLI (Semantic Natural Language Inference) dataset and demonstrates how fine-tuning the ELECTRA-small model with gender-neutralized data can mitigate occupational gender bias.

## Problem

NLP datasets collected through crowdsourcing often contain gender biases, particularly stereotypical associations between occupations and genders. These biases can lead to:
- Skewed predictions in NLI tasks
- Perpetuation of harmful stereotypes
- Reduced model generalization on gender-related tasks

## Methodology

**Bias Detection:**
- Applied the "competency problems" framework to identify artifact tokens in SNLI
- Used statistical analysis (z-scores with Bonferroni correction) to detect gender-biased patterns
- Analyzed multiple tokenization methods (unigrams, bigrams, custom "Bigram NS-Gen")
- Found that 32.4% of SNLI dataset artifacts were gender-related

**Bias Mitigation:**
- Created gender-neutralized training data by gender-swapping occupations (~30,000 sentences)
- Fine-tuned ELECTRA-small on combined SNLI + neutralized dataset
- All hypothesis pairs labeled as entailment to reduce bias

## Key Findings

**Dataset Artifacts:**
- Strong occupational stereotypes: "nanny" → female (73% → 0.3% contradiction probability shift when gender swapped)
- Identified three bias classes: strong associations, weaker associations, and neutral occupations
- Gender artifacts present across all label types (entailment, neutral, contradiction)

**Evaluation Metrics:**
- **False Negative Rate**: Reduced from 20% → 8% (male), 92% → 82% (female)
- **Model Bias (Unigrams)**: Decreased from 17% → 4.1% (entailment), 13.7% → 12.3% (contradiction)
- **Validation Accuracy**: Improved from 40.5% → 53.6% on gender-neutral test set
- **SNLI Performance**: Maintained 89.9% accuracy on original SNLI validation set

## Results

✅ Successfully reduced gender bias while maintaining model performance  
✅ Improved generalization on gender-related NLI tasks  
✅ Reduced false negative rates for both male and female categories  
✅ Decreased mean absolute difference in entailment probabilities for stereotypically biased occupations

## Limitations & Future Work

- Some heavily biased occupations still difficult to fully debias
- Overall MAD metric slightly increased (0.57 → 0.61)
- Future work: individual pronoun analysis, reversed premise/hypothesis evaluation, metrics accounting for all three labels

## Technologies

- Model: ELECTRA-small
- Dataset: SNLI + gender-neutralized augmentation
- Framework: Competency problems analysis with statistical significance testing



# fp-dataset-artifacts

## Getting Started
You'll need Python >= 3.6 to run the code in this repo.

First, clone the repository:

`git clone git@github.com:gregdurrett/fp-dataset-artifacts.git`

Then install the dependencies:

`pip install --upgrade pip`

`pip install -r requirements.txt`

If you're running on a shared machine and don't have the privileges to install Python packages globally,
or if you just don't want to install these packages permanently, take a look at the "Virtual environments"
section further down in the README.

To make sure pip is installing packages for the right Python version, run `pip --version`
and check that the path it reports is for the right Python interpreter.

## Training and evaluating a model
To train an ELECTRA-small model on the SNLI natural language inference dataset, you can run the following command:

`python3 train_electra.py --do_train --task nli --dataset snli --output_dir ./trained_model/`

Checkpoints will be written to sub-folders of the `trained_model` output directory.
To evaluate the final trained model on the SNLI dev set, you can use

`python3 train_electra.py --do_eval --task nli --dataset snli --model ./trained_model/ --output_dir ./eval_output/`

To prevent `run.py` from trying to use a GPU for training, pass the argument `--no_cuda`.

To train/evaluate a question answering model on SQuAD instead, change `--task nli` and `--dataset snli` to `--task qa` and `--dataset squad`.

**Descriptions of other important arguments are available in the comments in `run.py`.**

Data and models will be automatically downloaded and cached in `~/.cache/huggingface/`.
To change the caching directory, you can modify the shell environment variable `HF_HOME` or `TRANSFORMERS_CACHE`.
For more details, see [this doc](https://huggingface.co/transformers/v4.0.1/installation.html#caching-models).

An ELECTRA-small based NLI model trained on SNLI for 3 epochs (e.g. with the command above) should achieve an accuracy of around 89%, depending on batch size.
An ELECTRA-small based QA model trained on SQuAD for 3 epochs should achieve around 78 exact match score and 86 F1 score.

## Working with datasets
This repo uses [Huggingface Datasets](https://huggingface.co/docs/datasets/) to load data.
The Dataset objects loaded by this module can be filtered and updated easily using the `Dataset.filter` and `Dataset.map` methods.
For more information on working with datasets loaded as HF Dataset objects, see [this page](https://huggingface.co/docs/datasets/process.html).

## Virtual environments
Python 3 supports virtual environments with the `venv` module. These will let you select a particular Python interpreter
to be the default (so that you can run it with `python`) and install libraries only for a particular project.
To set up a virtual environment, use the following command:

`python3 -m venv path/to/my_venv_dir`

This will set up a virtual environment in the target directory.
WARNING: This command overwrites the target directory, so choose a path that doesn't exist yet!

To activate your virtual environment (so that `python` redirects to the right version, and your virtual environment packages are active),
use this command:

`source my_venv_dir/bin/activate`

This command looks slightly different if you're not using `bash` on Linux. The [venv docs](https://docs.python.org/3/library/venv.html) have a list of alternate commands for different systems.

Once you've activated your virtual environment, you can use `pip` to install packages the way you normally would, but the installed
packages will stay in the virtual environment instead of your global Python installation. Only the virtual environment's Python
executable will be able to see these packages.
"# nlp-final" 
