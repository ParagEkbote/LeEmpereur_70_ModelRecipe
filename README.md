## Model Recipe
This repository hosts the  model recipe of the model.

## Steps

### 1. First compute the block/layer similarity given a dataset

Navigate to the `compute_block_similarity` directory to run the layer similarity computation script.

1. Open `run_layer_similarity.sh`.
2. Replace the placeholder arguments with your model and dataset information:

```bash
python layer_similarity.py --model_path "mistralai/Mistral-7B-Instruct-v0.2" \
                      --dataset "arcee-ai/sec-data-mini" \
                      --dataset_column "text" \
                      --batch_size 8 \
                      --max_length 1024 \
                      --layers_to_skip 8 \
                      --dataset_size 4000 \
                      --dataset_subset "train" 
```

3. Execute the script to get the most optimum layer/block range to prune.

### Create the new model with the [Mergekit](https://github.com/arcee-ai/mergekit)

After identifying the layers to prune, you can proceed to the `slice_with_mergekit` directory and prune the model with the MergeKit library.
