//! Persistent map of (FEN, game_id) pairs to the highest epoch they were trained in.
//!
//! Used by the training loop to skip positions that have already been trained
//! at a higher epoch than the current one.

#![allow(clippy::cast_possible_truncation)]

use crate::persist::Persist;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub struct PositionDb {
    map: HashMap<u64, usize>,
}

impl PositionDb {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Derives the DB file path from the model output path.
    /// E.g. `"model.bin"` → `"model.bin.posdb"`
    pub fn db_path(output: &Path) -> PathBuf {
        let mut p = output.to_owned();
        p.as_mut_os_string().push(".posdb");
        p
    }

    /// Returns `true` if this (position, game) pair should be skipped for `current_epoch`.
    pub fn should_skip(&self, fen: &str, game_id: u64, current_epoch: usize) -> bool {
        self.map
            .get(&position_key(fen, game_id))
            .copied()
            .unwrap_or(0)
            >= current_epoch
    }

    /// Records that `(fen, game_id)` was trained at `epoch`. Only updates if
    /// the new epoch is higher than the stored value.
    pub fn record(&mut self, fen: &str, game_id: u64, epoch: usize) {
        let entry = self.map.entry(position_key(fen, game_id)).or_insert(0);
        if epoch > *entry {
            *entry = epoch;
        }
    }
}

impl Default for PositionDb {
    fn default() -> Self {
        Self::new()
    }
}

impl Persist for PositionDb {
    fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        w.write_all(&(self.map.len() as u64).to_le_bytes())?;
        for (&key, &epoch) in &self.map {
            w.write_all(&key.to_le_bytes())?;
            w.write_all(&(epoch as u64).to_le_bytes())?;
        }
        Ok(())
    }

    fn read_from<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let mut buf8 = [0u8; 8];

        r.read_exact(&mut buf8)?;
        let n = u64::from_le_bytes(buf8) as usize;
        let mut map = HashMap::with_capacity(n);

        for _ in 0..n {
            r.read_exact(&mut buf8)?;
            let key = u64::from_le_bytes(buf8);

            r.read_exact(&mut buf8)?;
            let epoch = u64::from_le_bytes(buf8) as usize;

            map.insert(key, epoch);
        }

        Ok(Self { map })
    }
}

// ─── hashing ─────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash over an arbitrary byte iterator.
///
/// Deterministic across runs — safe to use for persisted keys.
pub(crate) fn fnv1a(bytes: impl Iterator<Item = u8>) -> u64 {
    bytes.fold(14_695_981_039_346_656_037_u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(1_099_511_628_211)
    })
}

/// Combines a FEN string and a game ID into a single 64-bit map key.
fn position_key(fen: &str, game_id: u64) -> u64 {
    fnv1a(fen.bytes().chain(game_id.to_le_bytes()))
}
