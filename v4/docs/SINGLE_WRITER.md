# Store ownership and retry guarantees

`MemoryV4` holds an exclusive writer lease for the lifetime of the engine,
starting before any stored state is loaded. A second engine using the same
canonical data directory is refused, both in one process and across processes.
Use one `memory-os-v4 --transport http` server for several MCP clients; several
stdio processes must use separate stores or connect through that shared server.
The advisory lock requires a filesystem that supports OS file locks. Direct
low-level `Store`/`BlockStore` access does not acquire the facade's lease.

Call `engine.close()` when finished, or use `with MemoryV4(...) as engine:`.
Close flushes caches and releases the lease, including if the flush fails.
Failed initialization releases the lease too. Engines reject facade calls after
close and engines inherited through `fork` reject use in the child. Process exit
releases the OS lock; `writer.lock` remains on disk and must not be deleted.
Server shutdown closes its engine. Garbage collection releases a discarded
engine's lease as a fallback, without promising to flush its caches.

The mutating MCP tools (`memory_chat_save`, `memory_add_facts`,
`memory_block_set`, `memory_forget`, `memory_ingest`, `memory_migrate_v3`,
`memory_consolidate` and its `memory_compact` alias) accept an optional
`request_id`. Retries with the same ID, resolved scope, operation and validated
payload return the original completed response, including after restart.
Concurrent requests are serialized around the entire handler. Reusing an ID in
the same scope with a different operation or payload explicitly fails. Project
scopes remain distinct, including an all-project `*` scope; global block
requests use one global scope independent of the server's default project.
Calls without `request_id` retain their existing behavior.

`requests.jsonl` records a synced pending reservation before the handler runs.
After a successful handler, authoritative files (`items.jsonl`, `blocks.jsonl`,
`meta.json`) are synced before the completed receipt is synced and the
response returned. Directory entries are also synced on POSIX; Windows has no
directory-sync guarantee in this implementation. Derived indexes can be reconstructed at
startup. Completed receipts contain the original response, so protect and back
up the journal with the store. Journal corruption fails startup explicitly;
a failed journal write stops further ID-bearing writes in that process.

This is **not a crash-atomic transaction** across JSONL files. A crash or error
can leave a partly applied mutation and a pending request. Retrying that ID
returns an explicit incomplete-outcome error and never automatically reruns it.
Inspect the resulting items/blocks before deliberately submitting a new ID.
There is no automatic rollback, receipt expiration or compaction in this first
implementation. Atomic transactional storage and recovery belong to T08.
