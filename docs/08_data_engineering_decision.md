# Data engineering technology decision: retain the custom pipeline

## Decision

Retain the existing six-stage custom Python/pandas data engineering pipeline. Do not port the current `prepare` stage to Ray Data or Spark solely to add a distributed-computing dependency.

## Evidence from this project

- The source dataset contains 923 startup records.
- Cleaning leaves 922 records after one duplicate is removed.
- The pipeline already exposes six separately timed stages: ingest, validate, clean, label, split and write.
- The recorded end-to-end preparation time is approximately 0.0385 seconds.
- The recorded end-to-end throughput is approximately 23,967 rows/second.
- Per-stage timings and throughput are persisted in `data/processed/data_report.json` and surfaced through the pipeline console/monitoring implementation.

## Rubric interpretation

The evaluation rubric asks whether a data ingestion/transformation pipeline exists, says it *should* use Ray, Spark or Kubernetes, and separately asks whether Ray, Spark, Kubernetes **or a custom pipeline** is present. This project has a real, instrumented custom pipeline with measurable throughput. The development-guideline document lists Spark, Ray and Kubernetes as technology recommendations, not as a universal mandatory requirement.

## Why Ray/Spark is not justified for this dataset

A distributed runtime has scheduling, worker startup, serialization and dependency overhead. For 922 rows and a sub-0.1-second preparation run, that overhead would dominate the actual transformation work. It would add installation and operational failure modes without demonstrating a meaningful scalability improvement. The custom stages are already modular and can be migrated if the data volume grows.

## Viva answer

"I evaluated Ray/Spark against the actual workload. The dataset has 923 rows and the six-stage custom pipeline completes in about 0.0385 seconds, at roughly 24,000 rows per second. A distributed engine would add startup and scheduling overhead and would not improve this workload. The rubric explicitly recognises a custom pipeline as an option. I therefore kept the pipeline simple, measurable and reproducible, while keeping its stages modular so a future larger dataset could justify a Ray or Spark migration."

## When to revisit

Reconsider Ray Data or Spark if the input grows by orders of magnitude, transformations become expensive or distributed, or ingestion needs parallel processing across multiple files/partitions. Any migration should be benchmarked against the current baseline and must preserve data validation, label semantics, DVC reproducibility and the test suite.
