---
name: libaom/av1/separate-stage-multithreading-state
description: When adding encoder multi-threading state, isolate persistent stage state in dedicated structs and scope ephemeral synchronization and progress variables locally
rubric: design
severity: should_fix
---

**Rule.** When adding multi-threading state or synchronization callbacks for
distinct encoder stages, isolate persistent state in dedicated stage-specific
structs and AV1_COMP/AV1_PRIMARY members rather than general row multi-threading
structures (such as AV1EncRowMultiThreadInfo). Additionally, do not store
ephemeral synchronization structures, buffers, or transient error-recovery
progress counters (such as prepared worker counts) on persistent context structs
(like AV1_PRIMARY, AV1_COMP, or PrimaryMultiThreadInfo); declare them as local
variables within the executing function and pass them as parameters to cleanup
or restore helpers.

**Why.** Encapsulating persistent stage-specific multi-threading state in
dedicated structures avoids coupling independent pipeline passes and prevents
monolithic context structs. Scoping ephemeral synchronization data and
single-call progress state locally minimizes persistent struct size, prevents
context state pollution, and clarifies variable lifetime and error-recovery
ownership.

## Examples

Bad:

```cpp
// 1. Mixing stage-specific MT callbacks into general encoder row-MT info:
typedef struct {
  void (*intra_sync_read_ptr)(AV1EncRowMultiThreadSync *const, int, int);
} AV1EncRowMultiThreadInfo;

// 2. Attaching ephemeral sync structures or transient progress state to persistent structs:
typedef struct PrimaryMultiThreadInfo {
  AV1EncRowMultiThreadSync intra_row_mt_sync;
  int num_fpmt_workers_prepared;
} PrimaryMultiThreadInfo;

void prepare_fpmt_workers(AV1_PRIMARY *ppi, ...) {
  ppi->p_mt_info.num_fpmt_workers_prepared = i;
}
```

Good:

```cpp
// 1. Dedicated stage-specific struct and AV1_COMP member for persistent MT state:
typedef struct {
  void (*intra_sync_read_ptr)(AV1EncRowMultiThreadSync *const, int, int);
} AV1EncAllIntraMultiThreadInfo;

struct AV1_COMP {
  AV1EncRowMultiThreadInfo enc_row_mt;
  AV1EncAllIntraMultiThreadInfo intra_mt;
};

// 2. Declaring ephemeral sync structures and progress counters locally:
void prepare_fpmt_workers(AV1_PRIMARY *ppi, ...) {
  int num_fpmt_workers_prepared = 0;
  if (setjmp(error->jmp)) {
    restore_workers_after_fpmt(ppi, parallel_frame_count, num_fpmt_workers_prepared);
    ...
  }
  ...
  num_fpmt_workers_prepared = i;
}
```
