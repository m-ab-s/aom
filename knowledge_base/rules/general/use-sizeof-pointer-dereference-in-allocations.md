---
name: libaom/use-sizeof-pointer-dereference-in-allocations
description: When sizing allocations and memory operations with sizeof, use sizeof(*ptr), matching index dereferences, or sizeof(array) rather than explicit types, manual multiplications, or mismatched indices
rubric: readability
severity: blocking
---

**Rule.** When sizing dynamic allocations (aom_malloc, aom_calloc, aom_memalign,
aom_realloc, malloc, calloc), vector containers (aom_vector_setup), or memory
operations (memset, memcpy, memmove), derive element sizes using dereferenced
target expressions (such as sizeof(*ptr)) rather than hardcoding sizeof(Type).
For fixed-size local/static arrays or struct array members (e.g.,
ptr->member_array), pass sizeof(array) directly instead of multiplying element
size by dimension constants. When indexing into arrays of pointers or
multi-dimensional lookups, dereference the exact matching target expression
(such as sizeof(*arr[i]) or sizeof(*table[key])) or a clean base index
sizeof(*buf[0]), avoiding mismatched index variables or offsets (such as
sizeof(*dst->buf[c]) when copying dst->buf[dst_offset + c]).

**Why.** Deriving buffer and allocation sizes directly from the target pointer
(sizeof(*ptr)) or fixed array symbol (sizeof(array)) ensures sizes automatically
update when variable types or array dimensions change during refactoring,
preventing truncation, buffer overflows, and heap corruption. It also eliminates
redundant dimension constants and verbose arithmetic. Using exact matching
destination expressions or clean base index [0] prevents misleading indexing
mismatches and copy-paste errors.

## Examples

Bad:

```cpp
// Hardcoded type in allocation:
flow->u = (double *)aom_calloc(flow_size, sizeof(double));
block_hash_values[k] = (uint32_t *)aom_malloc(pic_size * sizeof(uint32_t));

// Manual element multiplication on fixed-size arrays:
int64_t M_trn[25];
memset(M_trn, 0, sizeof(*M_trn) * 25);
memset(gf_group->is_frame_dropped, 0,
       sizeof(gf_group->is_frame_dropped[0]) * MAX_STATIC_GF_GROUP_LENGTH);
memcpy(dst->stats, src->stats, sizeof(FIRSTPASS_STATS) * MAX_STATS_LEN);

// Mismatched index variable or dummy index in indexed operations:
memcpy(dst->buf[dst_offset + c], src->buf[c],
       sizeof(*dst->buf[c]) * src->width * src->height);
table[val] = aom_malloc(sizeof(table[0][0]));
```

Good:

```cpp
// Sizing derived from target pointer dereference:
CHECK_MEM_ERROR(cm, flow->u,
                (double *)aom_calloc(flow_size, sizeof(*flow->u)));
block_hash_values[k] =
    (uint32_t *)aom_malloc(pic_size * sizeof(*block_hash_values[k]));

// Fixed-size array size used directly:
int64_t M_trn[25];
memset(M_trn, 0, sizeof(M_trn));
memset(gf_group->is_frame_dropped, 0, sizeof(gf_group->is_frame_dropped));
memcpy(dst->stats, src->stats, sizeof(dst->stats));

// Consistent matching index or clean base index [0]:
memcpy(dst->buf[dst_offset + c], src->buf[c],
       sizeof(*dst->buf[0]) * src->width * src->height);
table[val] = aom_malloc(sizeof(*table[val]));
```
