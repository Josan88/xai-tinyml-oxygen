# 0003 - XAI Faithfulness Evaluation via Deletion and Insertion AUC

Status 11 September 2026: host evaluator has been run on all 52,090 canonical windows. Results and limits: [xai-faithfulness-aopc.md](../xai-faithfulness-aopc.md). That output is in-sample mean-replacement masking of the exported TinyKAN, not clinical or causal validation.

We evaluate local feature attribution faithfulness using Deletion and Insertion Area Over Perturbation Curves (AOPC / AUC) across all 52,090 OpenOximetry windows rather than ROAR retraining or synthetic ground truth. Progressively masking top-ranked features in normalized space measures whether Mean-Perturbation ranks change exported TinyKAN P faster than reverse or random orders. It does not identify clinical low-oxygen causes.
