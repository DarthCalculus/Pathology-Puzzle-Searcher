#pragma once
/*
 * Binary on-disk format for backsearch --harvest output.
 *
 * Layout:
 *   <HarvestFileHeader (64 B)>
 *   <argv blob — NUL-separated, length = header.argv_blob_len bytes>
 *   <HarvestRecord (144 B)> × N
 *
 * Endianness:
 *   All integers little-endian.  We target macOS/Linux on x86_64 + arm64.
 *   No cross-endianness portability is provided.
 *
 * Reading from Python:
 *   See 5x5_backsearch/harvest_load.py — uses np.frombuffer with the
 *   matching np.dtype.
 *
 * Versioning:
 *   On any layout change (record size or field meaning), bump HARVEST_VERSION
 *   and add a backwards-compatible reader.  Loaders MUST check magic and
 *   version before parsing.
 */

#include <stdint.h>

#define HARVEST_MAGIC       "BSH1"
#define HARVEST_VERSION     1

/* These match MAX_BLOCKS / MAX_HOLES in sokoban_bfs.h.  They are baked
 * into the wire format; changing them requires a version bump. */
#define HARVEST_MAX_BLOCKS  32
#define HARVEST_MAX_HOLES   32

/* Header flag bits. */
#define HARVEST_FLAG_ALLOW_EXIT_TRANSIT  (1u << 0)
#define HARVEST_FLAG_TWO_TABLES          (1u << 1)
#define HARVEST_FLAG_HOLELESS            (1u << 2)

/* One-shot file header. */
typedef struct __attribute__((packed)) {
    char     magic[4];               /* "BSH1" */
    uint16_t version;                /* HARVEST_VERSION */
    uint8_t  grid_rows;
    uint8_t  grid_cols;
    int16_t  exit_pos;               /* -1 if multi-exit run */
    uint16_t flags;                  /* HARVEST_FLAG_* */
    uint32_t reserved_a;
    uint64_t started_at_unix;        /* time(NULL) at harvest open */
    uint8_t  code_sha[20];           /* git rev-parse HEAD bytes; zero if unknown */
    uint32_t argv_blob_len;          /* bytes of NUL-separated argv following the header */
    uint8_t  reserved_b[16];
} HarvestFileHeader;                 /* sizeof == 64 */

/* Outcome codes: emitted as a single byte. */
#define HARVEST_OUTCOME_ACCEPTED        'A'
#define HARVEST_OUTCOME_SHORTCUT        'S'  /* shortcut_check returned a value >= 0 */
#define HARVEST_OUTCOME_SOLVER_ERROR    'E'  /* shortcut_check returned -2 (heap overflow) */
#define HARVEST_OUTCOME_DEDUP           'D'
#define HARVEST_OUTCOME_WALLS_CAP       'W'
#define HARVEST_OUTCOME_DEPTH_CAP       'X'

/* Per-state record.  Fixed 144 B for mmap + np.frombuffer reads. */
typedef struct __attribute__((packed)) {
    uint64_t state_id;                                  /* 0-7   */
    int64_t  parent_id;                                 /* 8-15  */
    uint64_t canonical_key;                             /* 16-23 */
    int32_t  depth;                                     /* 24-27 */
    int32_t  forward_solve;                             /* 28-31; -1 not called or no shortcut, -2 heap overflow, -99 unset */
    uint8_t  outcome;                                   /* 32 */
    uint8_t  nblocks;                                   /* 33 */
    uint8_t  nholes;                                    /* 34 */
    uint8_t  player_pos;                                /* 35 */
    int16_t  exit_pos;                                  /* 36-37 */
    uint8_t  reserved[2];                               /* 38-39 */
    uint64_t committed_empty;                           /* 40-47 */
    int8_t   block_pos[HARVEST_MAX_BLOCKS];             /* 48-79 */
    uint8_t  block_mask[HARVEST_MAX_BLOCKS];            /* 80-111 */
    int8_t   hole_pos[HARVEST_MAX_HOLES];               /* 112-143 */
} HarvestRecord;                                        /* sizeof == 144 */

/* Static asserts: catch silent layout drift at compile time.  Compilers
 * without _Static_assert will trip an array-size error instead. */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(HarvestFileHeader) == 64, "HarvestFileHeader must be 64 bytes");
_Static_assert(sizeof(HarvestRecord) == 144,    "HarvestRecord must be 144 bytes");
#else
typedef char _hfh_size_check[sizeof(HarvestFileHeader) == 64 ? 1 : -1];
typedef char _hr_size_check [sizeof(HarvestRecord)     == 144 ? 1 : -1];
#endif
