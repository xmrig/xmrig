This document describes the architectural differences and possible code changes required to integrate Junocash into XMRig. This is not all inclusive. I might be (probably am) missing things.

# SUPPORT_JUNOCASH

Guidelines for adding **Junocash** support to XMRig.

Junocash is a **Zcash fork** that uses **RandomX** PoW with a Zcash-style block header and RPC API. XMRig is currently oriented around **CryptoNote / Monero-style** coins and pools. Supporting Junocash means reusing XMRig's RandomX backend, but replacing all **coin-specific** pieces (block template parsing, header layout, RPC fields, nonce handling, and submit path) with the logic already implemented in the standalone **`juno-miner`** included in this repository.

---

## 1. Scope and Constraints

### 1.1 What this document covers

- Solo mining against a **local Junocash node** via `getblocktemplate` / `submitblock`.
- Using XMRig's existing **RandomX CPU backend** for hashing work.
- Adapting XMRig's **daemon client / job handling** to the Junocash block model.

### 1.2 What is explicitly _out of scope_ (for now)

- Pool mining with a Junocash-specific stratum protocol.
- GPU mining (RandomX is CPU-oriented; XMRig’s RandomX GPU support is separate).
- Full Zcash address parsing, shielding, or wallet features (the node handles coinbase construction).

The goal is: **“XMRig can mine Junocash blocks directly from a local node, using RandomX, with correctness equivalent to `juno-miner`.”**

---

## 2. High-Level Architecture Differences

### 2.1 XMRig today (Monero / CryptoNote)

Key components:

- **Algorithm selection** (`src/base/crypto/Algorithm.h`)
- **Coin-specific template parsing** (`src/base/tools/cryptonote/BlockTemplate.*`)
- **Job representation** (`src/base/net/stratum/Job.*`)
- **DaemonClient** for solo mining (`src/base/net/stratum/DaemonClient.*`)
- **RandomX backend** (`src/crypto/rx/*`) used for `rx/*` algorithms

Assumptions baked into this code:

- Block templates are **CryptoNote-style** (Monero): miner tx, extra fields, offsets into a blob, 4‑byte nonce.
- RPC fields: `blocktemplate_blob`, `blockhashing_blob`, `seed_hash`, etc.
- Nonce length is 4 bytes, tracked by `Job::nonceOffset()` and related helpers.

### 2.2 Junocash (as implemented in [juno-miner](https://github.com/juno-cash/juno-miner))

From `juno-miner/src/miner.h` and `miner.cpp`:

- The mining header is **140 bytes**:
  - 108 bytes: `CEquihashInput` (Zcash-style header without nonce)
    - `nVersion` (4 bytes, little-endian)
    - `hashPrevBlock` (32 bytes, internal order = little-endian storage)
    - `hashMerkleRoot` (32 bytes, internal order)
    - `hashBlockCommitments` (32 bytes, internal order)
    - `nTime` (4 bytes, little-endian)
    - `nBits` (4 bytes, little-endian)
  - 32 bytes: `nNonce` (full 256-bit nonce)
- Block template fields from RPC (see `parse_block_template`):
  - `version`
  - `previousblockhash` (hex, display order)
  - `curtime`
  - `bits` (hex string, compact target)
  - `height`
  - `randomxseedheight`
  - `randomxseedhash`
  - `randomxnextseedhash` (optional)
  - `defaultroots.merkleroot`
  - `defaultroots.blockcommitmentshash`
  - `coinbasetxn.data`
  - `transactions[i].data`
- RandomX seed:
  - Seed is the **32‑byte `randomxseedhash`** from the template.
  - Epoch behavior is managed by the node (`randomxseedheight`, `randomxnextseedhash`); the miner only needs to reinit when the seed changes.
- Nonce handling:
  - A full **256‑bit nonce** is appended to the 108‑byte header.
  - Threads use /dev/urandom to seed 256 bits, clear a few bits to avoid overlap, and then **increment the full 256‑bit value**.
  - Hash meets target if `hash <= target` for the 256‑bit target derived from `bits`.

These semantics differ significantly from the CryptoNote path hard-coded in XMRig’s `BlockTemplate` and `Job`.

**Conclusion:** Junocash integration should treat Junocash as a **separate “coin type” using RandomX**, not as another CryptoNote coin.

---

## 3. Integration Strategy in XMRig

### 3.1 Reuse vs replace

We will:

- **Reuse**:
  - RandomX backend (`src/crypto/rx/*`)
  - Threading, NUMA handling, and general job scheduling
  - Daemon HTTP / TLS handling infrastructure
- **Replace / bypass** for Junocash jobs:
  - CryptoNote `BlockTemplate` parsing (`src/base/tools/cryptonote/BlockTemplate.*`)
  - CryptoNote-specific submit path (hex-encoding nonce into `blocktemplate_blob`)
  - 4‑byte nonce assumptions in `Job` for daemon-based mining

### 3.2 Coin and algorithm wiring

1. **Add a Junocash algorithm ID (optional but recommended)**

   In `src/base/crypto/Algorithm.h`:

   ```cpp
   enum Id : uint32_t {
       // ... existing entries ...
       RX_YADA         = 0x72151279,   // "rx/yada"
+      RX_JUNO        = 0x7215126a,   // "rx/juno" (RandomX config same as rx/0)
   };
   ```

   - Family is still `RANDOM_X` so existing RandomX backend can be reused.
   - Parameters (L2/L3 sizes, iterations) must match the node’s RandomX config; if Junocash uses reference RandomX, map `RX_JUNO` to the same `RandomX_ConfigurationBase` as `RX_0`.

2. **Expose `rx/juno` in algorithm parsing / CLI**

   - Update `Algorithm::kRX_JUNO` string constant (similar to `kRX_WOW`, etc.) and parsing logic.
   - Ensure `--algo=rx/juno` and JSON configs like `{ "algo": "rx/juno" }` are accepted.

3. **Coin mapping**

   - If XMRig’s `Coin` class is used for daemon selection, add a `JUNO` entry that maps to `Algorithm::RX_JUNO` and indicates **Zcash-style** template, not CryptoNote.
   - Ensure pool/daemon configuration can specify `coin: "junocash"` or a similar identifier.

---

## 4. Junocash Block Template Parsing

We need a Junocash-specific block template parser modeled directly on `juno-miner`’s `BlockTemplate` struct and `parse_block_template` function in `juno-miner/src/miner.cpp`.

### 4.1 New data structure (JunocashTemplate)

Create new files, e.g.

- `src/base/tools/junocash/JunocashTemplate.h`
- `src/base/tools/junocash/JunocashTemplate.cpp`

Structure (modeled after `juno-miner::BlockTemplate`):

```cpp
struct JunocashTemplate {
    uint32_t version;
    std::string previous_block_hash;    // hex, display order
    std::string merkle_root;            // hex, display order
    std::string block_commitments_hash; // hex, display order
    uint32_t time;                      // nTime
    uint32_t bits;                      // compact target (nBits)
    std::vector<uint8_t> target;        // 32-byte target from bits
    std::string target_hex;             // optional, for logging/UI
    uint32_t height;

    uint64_t seed_height;
    std::vector<uint8_t> seed_hash;        // 32 bytes, internal order
    std::vector<uint8_t> next_seed_hash;   // optional

    std::vector<uint8_t> header_base;   // 140 bytes: 108 header + 32-byte zeroed nonce

    std::string coinbase_txn_hex;
    std::vector<std::string> txn_hex;   // non-coinbase tx hex strings
};
```

### 4.2 Parsing `getblocktemplate` (RapidJSON)

Implement `bool JunocashTemplate::parse(const rapidjson::Value &tpl)` equivalent to `parse_block_template` in `juno-miner/src/miner.cpp`:

Required fields (all present in `juno-miner`):

- `version` → `version`
- `previousblockhash` → `previous_block_hash`
- `curtime` → `time`
- `bits` (hex string) → `bits` (`std::stoul(bits, nullptr, 16)`)
- `height` → `height`
- `randomxseedheight` → `seed_height`
- `randomxseedhash` → `seed_hash` (hex → bytes, **NO reversal**, internal order already)
- Optional: `randomxnextseedhash` → `next_seed_hash` (hex → bytes, no reversal)
- `defaultroots.merkleroot` → `merkle_root`
- `defaultroots.blockcommitmentshash` (or top-level `blockcommitmentshash`) → `block_commitments_hash`
- `coinbasetxn.data` → `coinbase_txn_hex`
- `transactions[i].data` → `txn_hex.push_back(...)`

Derive additional fields:

- `target` = 256‑bit target derived from `bits` (use XMRig’s existing compact→target helper or copy `utils::compact_to_target` from `juno-miner/src/utils.cpp`).
- `target_hex` = hex string representation (for logging / debugging).

### 4.3 Header construction (Critical for correctness)

Use exactly the logic from `juno-miner::parse_block_template`:

1. Allocate `header_base` with size 140 bytes.
2. Serialize `CEquihashInput` into the first 108 bytes:

   ```cpp
   size_t offset = 0;

   // nVersion (4 bytes, little-endian)
   write_le32(&header_base[offset], version);
   offset += 4;

   // hashPrevBlock (32 bytes, internal order)
   auto prev = hex_to_bytes(previous_block_hash);   // 32 bytes
   std::reverse(prev.begin(), prev.end());          // convert display → internal
   std::copy(prev.begin(), prev.end(), header_base.begin() + offset);
   offset += 32;

   // hashMerkleRoot (32 bytes, internal order)
   auto merkle = hex_to_bytes(merkle_root);
   std::reverse(merkle.begin(), merkle.end());
   std::copy(merkle.begin(), merkle.end(), header_base.begin() + offset);
   offset += 32;

   // hashBlockCommitments (32 bytes, internal order)
   auto commits = hex_to_bytes(block_commitments_hash);
   std::reverse(commits.begin(), commits.end());
   std::copy(commits.begin(), commits.end(), header_base.begin() + offset);
   offset += 32;

   // nTime (4 bytes, little-endian)
   write_le32(&header_base[offset], time);
   offset += 4;

   // nBits (4 bytes, little-endian)
   write_le32(&header_base[offset], bits);
   offset += 4;

   assert(offset == 108);
   ```

3. Zero the **32‑byte nonce region** at `header_base[108..139]`:

   ```cpp
   std::fill(header_base.begin() + 108, header_base.end(), 0);
   ```

This must match `juno-miner` exactly; see `juno-miner/src/miner.cpp` for a proven implementation.

---

## 5. RandomX Integration for Junocash

Most RandomX details are already encapsulated in XMRig’s RandomX backend. Junocash only changes **where the seed comes from** and **how the header bytes are built**.

### 5.1 Seed handling

From `juno-miner`:

- Seed is always the **32‑byte `randomxseedhash`** from `getblocktemplate`.
- Epoch behavior (`randomxseedheight`, `randomxnextseedhash`) is node-managed; the miner just reinitializes when the seed hash changes.

In XMRig:

1. Extend `Job` (or a new Junocash-specific job type) to carry:

   - `std::array<uint8_t, 32> randomx_seed_hash;`
   - Optionally `uint64_t randomx_seed_height;`

2. In `DaemonClient::parseJob` (Junocash branch), map:

   ```cpp
   job.setSeedHash(hex_encode(junocashTemplate.seed_hash));
   job.setHeight(junocashTemplate.height);
   job.setDiff(difficulty_from_target(junocashTemplate.target));
   ```

3. Ensure the RandomX backend uses this `seed_hash` to manage cache/dataset reinitialization (similar to how it does for Monero’s `seed_hash`). You may reuse existing RandomX seed logic; only the source JSON field changes.

### 5.2 RandomX flags / modes

`juno-miner` uses:

```cpp
randomx_flags flags = randomx_get_flags();
flags |= RANDOMX_FLAG_JIT;

randomx_flags vm_flags = flags;
if (fast_mode_) {
    vm_flags |= RANDOMX_FLAG_FULL_MEM;
}
```

XMRig already configures RandomX VMs similarly. For Junocash:

- Use the **same flags** as `rx/0` unless the Junocash daemon uses a custom config.
- Honor XMRig’s existing `rx/` tuning options (NUMA, huge pages, fast vs light mode) for `RX_JUNO` as for `RX_0`.

---

## 6. Job Representation and Nonce Handling

Junocash uses a **32‑byte nonce** appended to a 108‑byte header. XMRig’s CryptoNote logic assumes a **4‑byte nonce** inside a larger blob.

### 6.1 Extending `Job` for Junocash

In `src/base/net/stratum/Job.*`:

1. Add fields for Junocash jobs:

   ```cpp
   class Job {
       // existing fields ...

       // Junocash-specific
       std::vector<uint8_t> m_junoHeaderBase;   // 140 bytes (108 header + 32-byte zeroed nonce)
       bool m_isJunocash = false;
   };
   ```

2. New setters / getters:

   ```cpp
   void setJunocashHeader(const std::vector<uint8_t> &headerBase) {
       m_junoHeaderBase = headerBase;
       m_isJunocash = true;
   }

   bool isJunocash() const { return m_isJunocash; }
   const std::vector<uint8_t> &junocashHeaderBase() const { return m_junoHeaderBase; }
   ```

3. For non-Junocash jobs, maintain existing CryptoNote behavior (`m_isJunocash == false`).

### 6.2 Mining loop for Junocash jobs

Instead of relying on CryptoNote `blob` + 4‑byte nonce offset, implement a Junocash-specific mining path using the same semantics as `Miner::worker_thread` in `juno-miner/src/miner.cpp`:

Pseudo-code for the hashing loop:

```cpp
void mineJunocash(const Job &job, randomx_vm *vm, ThreadState &state) {
    const auto &headerBase = job.junocashHeaderBase(); // 140 bytes

    std::vector<uint8_t> hash_input(140);
    std::copy(headerBase.begin(), headerBase.begin() + 108, hash_input.begin());

    // Initialize 256-bit nonce from /dev/urandom
    std::vector<uint8_t> nonce(32, 0);
    fill_nonce_from_urandom(nonce.data(), 32);

    // Clear bottom/top 16 bits (optional, matches juno-miner):
    nonce[0] = nonce[1] = 0;
    nonce[30] = nonce[31] = 0;

    uint8_t hash[32];

    while (!shouldStop) {
        // Copy nonce into bytes [108..139]
        std::copy(nonce.begin(), nonce.end(), hash_input.begin() + 108);

        randomx_calculate_hash(vm, hash_input.data(), hash_input.size(), hash);

        if (hash_meets_target(hash, job.targetBytes())) {
            // submit result: include full 32-byte nonce and full 140-byte header
            report_found_solution(job, nonce, hash_input, hash);
            break;
        }

        // Increment full 256-bit nonce (least significant byte first)
        increment_256bit_nonce(nonce);
    }
}
```

This is almost identical to `juno-miner` and ensures byte-for-byte parity with the reference implementation.

---

## 7. Daemon RPC Integration

### 7.1 `getblocktemplate`

Use XMRig’s existing `DaemonClient` HTTP machinery but **change the payload and parsing** when the coin is Junocash.

1. In `DaemonClient::getBlockTemplate()`:

   - For CryptoNote coins, current code sends:

     ```cpp
     params.AddMember("wallet_address", m_user.toJSON(), allocator);
     params.AddMember("extra_nonce", Cvt::toHex(...).toJSON(doc), allocator);
     JsonRequest::create(doc, m_sequence, "getblocktemplate", params);
     ```

   - For Junocash, construct the request to match what `juno-miner` sends (inspect `juno-miner/src/rpc_client.cpp` for exact fields). At minimum you will need:

     - The **mining address** (t‑address) configured in XMRig (map from existing `user`/`wallet` options).
     - Any Junocash-specific flags (`capabilities`, etc.), if required by the node.

   Keep this logic **coin-gated** so existing CryptoNote behavior is unchanged.

2. In `DaemonClient::parseJob`:

   - Detect Junocash coin:

     ```cpp
     if (m_coin.isJunocash()) {
         JunocashTemplate tpl;
         if (!tpl.parse(params)) {
             return jobError("Invalid Junocash block template");
         }

         Job job(false, Algorithm::RX_JUNO, String());
         job.setJunocashHeader(tpl.header_base);
         job.setHeight(tpl.height);
         job.setSeedHash(bytes_to_hex(tpl.seed_hash));
         job.setDiff(difficulty_from_target(tpl.target));

         // no CryptoNote-specific BlockTemplate parse here
         m_job = std::move(job);
         m_currentJobId = Cvt::toHex(Cvt::randomBytes(4));
         m_job.setId(m_currentJobId);

         m_prevHash    = nullptr; // or track blockhash if node returns it
         m_jobSteadyMs = Chrono::steadyMSecs();

         if (m_state == ConnectingState) {
             setState(ConnectedState);
         }

         m_listener->onJobReceived(this, m_job, params);
         return true;
     }

     // else: existing CryptoNote path
     ```

### 7.2 `submitblock`

Current CryptoNote implementation in `DaemonClient::submit`:

- Edits `m_blocktemplateStr` in-place: encodes nonce and optional miner signature at specific offsets inside `blocktemplate_blob`.
- Sends `submitblock([ blocktemplate_blob ])`.

For Junocash:

1. Build the full block hex according to Junocash node’s expectations. Reuse the logic from `juno-miner/src/rpc_client.cpp`:

   - Serialize the **140-byte header (including winning nonce)**
   - Append the coinbase and other transactions in the order required by the node
   - (If Junocash uses any extra fields beyond `nNonce` and transactions, mirror them exactly from `juno-miner`.)

2. Create a Junocash-specific submit path, e.g.:

   ```cpp
   int64_t DaemonClient::submitJunocash(const JobResult &result, const JunocashTemplate &tpl) {
       // 1. Rebuild full header (108 bytes from tpl.header_base + 32-byte winning nonce from result)
       // 2. Construct full block: header + coinbase_txn_hex + txn_hex[]
       // 3. JSON-RPC submitblock with one param: hex-encoded block
   }
   ```

3. In `DaemonClient::submit`, branch on `m_coin.isJunocash()` and call the Junocash-specific helper instead of the CryptoNote path.

---

## 8. Testing and Verification

To ensure correctness, leverage the existing `juno-miner` test utilities:

### 8.1 Block 1583 verification

`juno-miner/verify_block_1583.cpp` verifies that the RandomX hash over the constructed 140‑byte header matches a known block hash.

- After integrating Junocash into XMRig, add an equivalent test that:
  - Reads the same block template data (or hardcodes the values from `verify_block_1583.cpp`).
  - Builds the header using XMRig’s Junocash path.
  - Runs RandomX and compares to the known expected hash.

### 8.2 Template comparison tools

`juno-miner` contains several utilities:

- `test_hash_verification.cpp`
- `test_comparison.cpp`
- `test_simple_mine.cpp`
- `test_mining_simple.cpp`

They:

- Show the full 140‑byte header in hex.
- Break down the header by fields (version, prev hash, merkle root, commitments, time, bits).
- Compare template-derived headers to known block data.

You can:

- Use their output to **byte-compare** XMRig’s header construction.
- Port a minimal subset of these tests (or keep `juno-miner` alongside XMRig for developer verification).

### 8.3 End-to-end test plan

1. Start a local Junocash node with RPC enabled.
2. Run XMRig configured for Junocash (`--algo=rx/juno`, `--coin=junocash`, daemon host/port/user/pass).
3. Confirm:
   - XMRig successfully fetches `getblocktemplate` from the node.
   - XMRig logs show the same height, bits, seed hash as `juno-miner`.
   - For the same template, RandomX hashes computed by XMRig and `juno-miner` match.
   - When a solution is found, `submitblock` succeeds and the block is accepted.

---

## 9. Implementation Checklist

This is a practical step-by-step checklist to track Junocash support implementation.

1. **Algorithm & Coin Wiring**
   - [ ] Add `RX_JUNO` to `Algorithm::Id` and expose it via CLI/config (`rx/juno`).
   - [ ] If using `Coin` abstraction, add a `Junocash` entry mapping to `RX_JUNO` and mark it as Zcash-style.

2. **Junocash Template Parser**
   - [ ] Create `JunocashTemplate.{h,cpp}` in a suitable directory (e.g. `src/base/tools/junocash/`).
   - [ ] Implement parsing of all required RPC fields (`version`, `previousblockhash`, `curtime`, `bits`, `height`, `randomxseed*`, `defaultroots.*`, `coinbasetxn`, `transactions`).
   - [ ] Implement compact target → 256‑bit target conversion.
   - [ ] Implement header construction (140 bytes) exactly matching `juno-miner`.

3. **Job Extensions**
   - [ ] Extend `Job` to support Junocash jobs (`m_isJunocash`, `m_junoHeaderBase`, seed hash fields).
   - [ ] Provide setters/getters for Junocash header and seed.

4. **DaemonClient Integration**
   - [ ] In `getBlockTemplate()`, send Junocash-appropriate JSON-RPC params when `coin == Junocash`.
   - [ ] In `parseJob`, branch by coin:
       - [ ] CryptoNote coins: keep current `BlockTemplate`-based path.
       - [ ] Junocash: use `JunocashTemplate`, set `Job` fields, and skip CryptoNote `BlockTemplate`.
   - [ ] Store any additional Junocash-specific data needed for submission (e.g. transactions, if not kept in `JunocashTemplate`).

5. **Mining Loop Changes**
   - [ ] Implement a Junocash-specific RandomX hashing path that:
       - [ ] Uses the 140‑byte header + 32‑byte nonce layout.
       - [ ] Initializes and increments a full 256‑bit nonce.
       - [ ] Uses the 256‑bit target from `bits`.
   - [ ] Integrate this path into XMRig’s backend only when `job.isJunocash()` is true.

6. **Submit Path**
   - [ ] Implement `submitJunocash()` in `DaemonClient` (or similar helper):
       - [ ] Rebuild the full block hex using header + coinbase + transactions.
       - [ ] Call `submitblock` with this hex.
   - [ ] Wire `DaemonClient::submit` to call `submitJunocash()` for Junocash jobs.

7. **Testing & Validation**
   - [ ] Add a unit/integration test mirroring `verify_block_1583.cpp`.
   - [ ] Compare header bytes and RandomX hashes from XMRig and `juno-miner`.
   - [ ] Perform an end-to-end local node test and mine at least one real Junocash block.

---

## 10. References

- `juno-miner/src/miner.h` and `miner.cpp` – authoritative reference for:
  - Block template parsing
  - Header construction
  - RandomX seed handling
  - Nonce generation and increment
- `juno-miner/test_*` and `verify_block_1583.cpp` – reference for correctness tests.
- XMRig source:
  - `src/base/crypto/Algorithm.h` – algorithm IDs and families
  - `src/crypto/rx/*` – RandomX backend
  - `src/base/net/stratum/DaemonClient.*` – daemon solo mining
  - `src/base/tools/cryptonote/BlockTemplate.*` – CryptoNote block template handling (for comparison only)

With these changes, XMRig will have a clear, maintainable integration path for Junocash that mirrors the proven behavior of the standalone `juno-miner` while reusing XMRig’s mature RandomX backend and infrastructure.
```