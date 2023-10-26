[Giang]

- [How to run this code](#How-to-run-this-code)
- [How this code works](#How-this-code-works)
  
Few notes for this repository:

### How to run this code

- Machine I use to run:
   **10.0.104.137** at **/mnt/hdd2/gtnguyen/lm-watermarking**
  
- Run docker container for environments:
  ```docker run --name giang-llm-watermark -it --rm --shm-size 16G --gpus all -v /mnt/hdd2/gtnguyen/lm-watermarking:/work giang-img```

- Experiments are located at folder: **./experiments**

- Run command:
    ```python3 run_watermarking.py --model_name facebook/opt-1.3b --dataset_name c4 --dataset_config_name realnewslike --max_new_tokens 200 --min_prompt_tokens 50 --limit_indices 500 --input_truncation_strategy completion_length --input_filtering_strategy prompt_and_completion_length --output_filtering_strategy max_new_tokens --dynamic_seed markov_1 --bl_proportion 0.5 --bl_logit_bias 2.0 --bl_type soft --store_spike_ents True --num_beams 1 --use_sampling True --sampling_temp 0.7 --oracle_model_name facebook/opt-2.7b --run_name example_run --output_dir ./all_runs```


The parameters' name explains itself but here is few notes when running it:

- **param output_dir**: where the result is stored
  If the code is run successfully, there will be 3 files created: **gen_table_meta.json**, **gen_table_w_metrics.jsonl**, **gen_table.jsonl**

- About **wandb**, they use wandb to log, track and manage the experiment. There are 3 params related to wandb:

    **--no_wandb** : Whether to log to wandb.
  
    **--wandb_entity**: The wandb entity/user for the project. This is the username of your wandb account.
  
    **--wandb_project**: The name of the wandb project. So you create a project and set its name to this param. When you open this project on wandb, you will see what have been tracked

  You will need to create an access token on wandb and paste to the commandline once the script asks in the terminal.
  
  I see wandb track things like CPU Utilization, Memory Utilization, Network Traffic, etc. Overall, I don't think it's useful for me.

- Further information about the params/arguments can be read in run_watermarking.py of the same folder.

- While running the script, there will be times that you feel like the process is frozen (at least for the machine that I'm running). Just wait it to run, lol.

---------------

### How this code works (I also commented on the code):

- First, load the model (e.g. **facebook/opt-1.3b**)
  
- Load the dataset (e.g. **c4 realnewlikes**)
  
- Construct the blacklist processor/sampler that enforces that specified sequences will never be sampled. They create a class **BlacklistLogitsProcessor** (in **watermark.py**) that implements **LogitsProcessor** of transformers lib.

  Basically, a **LogitsProcessor** can be used to modify the prediction scores of a language model head for generation.
  
- Construct the pipeline for generation and measurement that pulls from the streaming dataset, applies the generations map funcs. Specifically, in **run_watermarking.py**, firstly, there is only the **dataset** instance.
  By applying functions (i.e., map(), shuffle(), filter()), they define the pipeline (but no execution yet). The pipeline is: indexed_dataset -> shuffled_dataset -> tokenized_and_truncated_dataset -> input_length_filtered_dataset -> generations_dataset.
  To execute, they create an iterator object using **generations_dataset** and call ```next()``` on the iterator
