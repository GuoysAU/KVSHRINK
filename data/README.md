# Datasets

## Expected layout

    data/
    ├── boolq/dev.jsonl
    ├── openbookqa/dev.jsonl
    ├── hellaswag/dev.jsonl
    ├── winogrande/dev.jsonl
    └── xsum/dev.jsonl

Each file holds one example per line, in the original order of the source
split. All evaluations in the paper use the first *N* lines of each file
(`--max-samples`), so the ordering matters for reproducing the reported numbers.

## Download

All five files are available here:

  https://drive.google.com/drive/folders/1CSb80vMVyiEjVbJJcZdkzvA47J-Qkpxj
