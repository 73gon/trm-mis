#!/bin/bash -l
# Parallel n=200 multi-MIS data build. 6 train workers (10k total) + 2 test
# workers (1k total). Each worker runs single-threaded Gurobi; the workers
# themselves run in parallel.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT=data/smallmis_n200_multi
mkdir -p "$OUT" logs

PIDS=()

# Train chunks: 6 workers * 2000 graphs = 12000 train. Seeds non-overlapping.
for i in 0 1 2 3 4 5; do
  seed_start=$((5000000 + i * 100000))
  out_dir="$OUT/train_chunk_$i"
  mkdir -p "$out_dir"
  .venv/bin/python -m dataset.build_mis_dataset_multilabel \
    --output-dir "$out_dir" \
    --num-instances 2000 \
    --shard-size 500 \
    --seed-start "$seed_start" \
    --n-min 200 --n-max 200 \
    --d-min 7.35 --d-max 7.35 \
    --pool-size 16 --pool-gap 0.0 --pool-time-limit 60.0 --threads 1 \
    > "logs/build_n200_train_chunk_$i.log" 2>&1 &
  PIDS+=($!)
done

# Test chunks: 2 workers * 500 graphs = 1000 test.
for i in 0 1; do
  seed_start=$((6000000 + i * 100000))
  out_dir="$OUT/test_chunk_$i"
  mkdir -p "$out_dir"
  .venv/bin/python -m dataset.build_mis_dataset_multilabel \
    --output-dir "$out_dir" \
    --num-instances 500 \
    --shard-size 500 \
    --seed-start "$seed_start" \
    --n-min 200 --n-max 200 \
    --d-min 7.35 --d-max 7.35 \
    --pool-size 16 --pool-gap 0.0 --pool-time-limit 60.0 --threads 1 \
    > "logs/build_n200_test_chunk_$i.log" 2>&1 &
  PIDS+=($!)
done

echo "Launched ${#PIDS[@]} workers: ${PIDS[*]}"
echo "${PIDS[*]}" > logs/build_n200_pids.txt
wait "${PIDS[@]}"
echo "All workers finished."
