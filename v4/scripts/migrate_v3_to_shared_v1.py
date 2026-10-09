"""One-shot (idempotent) migration of the v3 conversation log into the shared
v4 data dir used by the HTTP server. Run BEFORE starting the shared server:
the store must have a single writer."""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MEMORY_V4_DIR", os.path.expanduser("~/.memory-os-v4"))

from memory_os_v4.config import Config
from memory_os_v4.memory import MemoryV4

log = sys.argv[1] if len(sys.argv) > 1 else "~/.memory-os-ai/_conversation_log.jsonl"
t0 = time.time()
eng = MemoryV4(config=Config.from_env())
res = eng.migrate_v3(log)
eng.flush()
print(json.dumps({"data_dir": os.environ["MEMORY_V4_DIR"], "migrate": res,
                  "status": eng.status(), "seconds": round(time.time() - t0, 1)},
                 ensure_ascii=False, default=str, indent=1))
