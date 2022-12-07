# Incorporating Emotions into Health Mention Classification Task on Social Media

The code for the paper *Incorporating Emotions into Health Mention Classification Task on Social Media*.

## Dataset
Our proposed approach was evaluated on the list of datasets below. All dataset need to be downloaded from the respective providers.

1. `FLU2013` - *Separating Fact from Fear: Tracking Flu Infections on Twitter*. For more information, see http://michaeljpaul.com/downloads/flu_data.php
2. `PHM2017` - *Did You Really Just Have a Heart Attack? Towards Robust Detection of Personal Health Mentions in Social Media*. For more information, see https://github.com/emory-irlab/PHM2017
3. `HMC2019` - *Leveraging Sentiment Distributions to Distinguish Figurative From Literal Health Reports on Twitter*. For more information, see https://github.com/biddle-r/HMC2019
4. `SELF2020` -  *Identifying Medical Self-Disclosure in Online Communities* - For more information, contact, mvaliz2@uic.edu
5. `RHMD2022` - *Identification of Disease or Symptom terms in Reddit to Improve Health Mention Classification* - For more information, see https://github.com/usmaann/RHMD-Health-Mention-Dataset

## Basic Usage

### Preprocessing
```
python preprocess.py --data_path [path_to_data] \
                     --text_column [name_of_text_column] \
                     --label_column [name_of_label_column] \
                     --output_dir [path_to_save_processed_data]
```

### HMC experiments

**For intermediate task fine-tuning approach**
```
python run_phm.py   --train_file [train_file] \
                    --validation_file [validation_file] \
                    --test_file [test_file]\
                    --bert_model [path_to_emotion_model] \
                    --num_train_epochs [num_of_epochs] \
                    --per_device_train_batch_size [train_batch_size] \
                    --per_device_eval_batch_size [eval_batch_size] \
                    --model_type base

```
**For multi-feature fusion approach**
```
python run_phm.py   --train_file [train_file] \
                    --validation_file [validation_file] \
                    --test_file [test_file]\
                    --bert_model bert-base-uncased \
                    --emotion_model [path_to_emotion_model] \
                    --num_train_epochs [num_of_epochs] \
                    --per_device_train_batch_size [train_batch_size] \
                    --per_device_eval_batch_size [eval_batch_size] \
                    --model_type multi_feature

```
### Run finetune experiments
```
python run_finetune.py  --train_file [train_file] \
                        --validation_file [validation_file] \
                        --bert_model bert-base-uncased \
                        --num_train_epochs [num_of_epochs] \
                        --per_device_train_batch_size [train_batch_size] \
                        --per_device_eval_batch_size [eval_batch_size] \
                        --output_dir [path_to_save_model] \
                        --model_type base
```

## Requirements
```
python==3.9.6
torch==1.11.0
transformers==4.21.1
```
