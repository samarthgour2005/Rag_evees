"""One lock for every Hugging Face model load in the process.

transformers builds a model by patching ``torch.nn.Module.register_parameter``
so new parameters land on the meta device, then loads the weights in. The patch
is process-wide, not per-thread. When two loads overlap, one can save the
other's patched version as the "original" and put it back on exit, so every
later load in the process gets meta tensors and fails in ``model.to()`` with
"Cannot copy out of meta tensor" until the process restarts. On the deployment
that was the embedder and a reranker loading in two request threads at once.

Every model construction (embedder, cross-encoder, ColBERT) holds this lock.
Re-entrant, because ``shared_reranker`` holds it while ``build_reranker`` runs.
"""

import threading

MODEL_LOAD_LOCK = threading.RLock()
