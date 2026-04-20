//! Persistent map of FEN positions to the highest epoch they were trained in.
//!
//! Used by the training loop to skip positions that have already been trained
//! at a higher epoch than the current one.

#![allow(clippy::cast_possible_truncation)]

use crate::persist::Persist;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub struct PositionDb {
    map: HashMap<String, usize>,
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

    /// Returns `true` if this position should be skipped for `current_epoch`.
    ///
    /// Skip condition: `stored_epoch > current_epoch`.
    pub fn should_skip(&self, fen: &str, current_epoch: usize) -> bool {
        self.map.get(fen).copied().unwrap_or(0) > current_epoch
    }

    /// Records that `fen` was trained at `epoch`. Only updates if the new
    /// epoch is higher than the stored value.
    pub fn record(&mut self, fen: String, epoch: usize) {
        let entry = self.map.entry(fen).or_insert(0);
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
        for (fen, &epoch) in &self.map {
            let bytes = fen.as_bytes();
            w.write_all(&(bytes.len() as u64).to_le_bytes())?;
            w.write_all(bytes)?;
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
            let fen_len = u64::from_le_bytes(buf8) as usize;

            let mut fen_bytes = vec![0u8; fen_len];
            r.read_exact(&mut fen_bytes)?;
            let fen = String::from_utf8(fen_bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            r.read_exact(&mut buf8)?;
            let epoch = u64::from_le_bytes(buf8) as usize;

            map.insert(fen, epoch);
        }

        Ok(Self { map })
    }
}
