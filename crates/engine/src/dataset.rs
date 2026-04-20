//! `ChessDataset`: holds (Board, outcome) samples and provides shuffled mini-batches.

#![allow(clippy::cast_possible_truncation)]

use crate::fen_file::parse_fen_file;
use crate::persist::Persist;
use crate::pgn::{parse_pgn, Sample};
use chess::fen;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use tracing::{info, warn};

/// A dataset of (board position, outcome) pairs for supervised training.
pub struct ChessDataset {
    pub samples: Vec<Sample>,
}

impl ChessDataset {
    /// Creates an empty dataset.
    #[must_use]
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Loads and parses all games from a single PGN file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read.
    pub fn from_pgn(path: &Path) -> Result<Self, std::io::Error> {
        let text = std::fs::read_to_string(path)?;
        Ok(Self {
            samples: parse_pgn(&text),
        })
    }

    /// Loads positions from multiple files or directories.
    ///
    /// Supports `.pgn`, `.fen`, and `.epd` files.  Directories are scanned for
    /// all files with those extensions.  Unreadable files are skipped with a
    /// warning on stderr.
    #[must_use]
    pub fn from_pgn_files(paths: &[std::path::PathBuf]) -> Self {
        let mut ds = Self::new();
        for path in paths {
            if path.is_dir() {
                let entries = match std::fs::read_dir(path) {
                    Ok(e) => e,
                    Err(e) => {
                        warn!(path = %path.display(), error = %e, "cannot read directory");
                        continue;
                    }
                };
                let mut found: Vec<_> = entries
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| is_supported_extension(p))
                    .collect();
                found.sort();
                for p in &found {
                    ds.load_one(p);
                }
            } else {
                ds.load_one(path);
            }
        }
        ds
    }

    /// Derives the cache file path from the model output path.
    /// E.g. `"model.bin"` → `"model.bin.dscache"`
    #[must_use]
    pub fn cache_path(output: &Path) -> PathBuf {
        let mut p = output.to_owned();
        p.as_mut_os_string().push(".dscache");
        p
    }

    /// Loads from a binary cache if valid, otherwise parses PGN files and saves the cache.
    ///
    /// The cache is valid when it exists and is newer than every source file.
    #[must_use]
    pub fn load_with_cache(paths: &[PathBuf], cache_path: &Path) -> Self {
        if cache_is_valid(cache_path, paths) {
            match Self::load_from(cache_path) {
                Ok(ds) => {
                    info!(samples = ds.len(), "loaded dataset from cache");
                    return ds;
                }
                Err(e) => warn!(error = %e, "cache unreadable — re-parsing"),
            }
        }
        let ds = Self::from_pgn_files(paths);
        if let Err(e) = ds.save_to(cache_path) {
            warn!(error = %e, "could not save dataset cache");
        } else {
            info!(samples = ds.len(), "dataset cache written");
        }
        ds
    }

    /// Extends the dataset with additional samples (e.g. from self-play).
    pub fn extend(&mut self, samples: impl IntoIterator<Item = Sample>) {
        self.samples.extend(samples);
    }

    /// Number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns `true` if the dataset has no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Shuffles samples in-place using a deterministic LCG seeded by `seed`.
    pub fn shuffle(&mut self, seed: u64) {
        let n = self.samples.len();
        if n < 2 {
            return;
        }
        let mut rng = seed.wrapping_add(1);
        for i in (1..n).rev() {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let j = (rng >> 33) as usize % (i + 1);
            self.samples.swap(i, j);
        }
    }

    /// Iterates over non-overlapping mini-batches of `size`.
    ///
    /// The last batch may be smaller than `size`.
    pub fn batches(&self, size: usize) -> impl Iterator<Item = &[Sample]> {
        self.samples.chunks(size)
    }

    fn load_one(&mut self, path: &Path) {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let before = self.samples.len();
                let new_samples = if is_fen_extension(path) {
                    parse_fen_file(&text)
                } else {
                    parse_pgn(&text)
                };
                self.samples.extend(new_samples);
                info!(
                    samples = self.samples.len() - before,
                    path = %path.display(),
                    "loaded file"
                );
            }
            Err(e) => warn!(path = %path.display(), error = %e, "cannot read file"),
        }
    }
}

impl Default for ChessDataset {
    fn default() -> Self {
        Self::new()
    }
}

/// Binary cache format: `[u64 num_samples]` then for each sample:
/// `[u64 fen_len] [u8; fen_len] [f32 label]`
impl Persist for ChessDataset {
    fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        w.write_all(&(self.samples.len() as u64).to_le_bytes())?;
        for (board, label) in &self.samples {
            let fen_bytes = board.to_fen();
            let bytes = fen_bytes.as_bytes();
            w.write_all(&(bytes.len() as u64).to_le_bytes())?;
            w.write_all(bytes)?;
            w.write_all(&label.to_le_bytes())?;
        }
        Ok(())
    }

    fn read_from<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let mut buf8 = [0u8; 8];
        let mut buf4 = [0u8; 4];

        r.read_exact(&mut buf8)?;
        let n = u64::from_le_bytes(buf8) as usize;
        let mut samples = Vec::with_capacity(n);

        for _ in 0..n {
            r.read_exact(&mut buf8)?;
            let fen_len = u64::from_le_bytes(buf8) as usize;
            let mut fen_bytes = vec![0u8; fen_len];
            r.read_exact(&mut fen_bytes)?;
            let fen = String::from_utf8(fen_bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let board = fen::from_fen(&fen)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
            r.read_exact(&mut buf4)?;
            samples.push((board, f32::from_le_bytes(buf4)));
        }

        Ok(Self { samples })
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Returns `true` if the cache file exists and is newer than every source file.
fn cache_is_valid(cache_path: &Path, paths: &[PathBuf]) -> bool {
    let Ok(cache_mtime) = fs::metadata(cache_path).and_then(|m| m.modified()) else {
        return false;
    };
    match newest_source_mtime(paths) {
        None => false,
        Some(src) => cache_mtime.duration_since(src).is_ok(),
    }
}

/// Returns the newest modification time across all source files (recursing one
/// level into directories).
fn newest_source_mtime(paths: &[PathBuf]) -> Option<SystemTime> {
    let mut newest: Option<SystemTime> = None;
    let mut consider = |path: &Path| {
        if let Ok(mtime) = fs::metadata(path).and_then(|m| m.modified()) {
            newest = Some(match newest {
                None => mtime,
                Some(n) => if mtime > n { mtime } else { n },
            });
        }
    };
    for path in paths {
        if path.is_dir() {
            if let Ok(entries) = fs::read_dir(path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if is_supported_extension(&p) {
                        consider(&p);
                    }
                }
            }
        } else {
            consider(path);
        }
    }
    newest
}

fn is_fen_extension(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("fen") || ext.eq_ignore_ascii_case("epd"))
}

fn is_supported_extension(path: &Path) -> bool {
    path.extension().is_some_and(|ext| {
        ext.eq_ignore_ascii_case("pgn")
            || ext.eq_ignore_ascii_case("fen")
            || ext.eq_ignore_ascii_case("epd")
    })
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use chess::board::Board;

    fn dummy_samples(n: usize) -> Vec<Sample> {
        (0..n)
            .map(|_| (Board::starting_position(), 0.0f32))
            .collect()
    }

    #[test]
    fn batches_split_correctly() {
        let mut ds = ChessDataset::new();
        ds.extend(dummy_samples(10));
        let batches: Vec<_> = ds.batches(3).collect();
        assert_eq!(batches.len(), 4); // 3 + 3 + 3 + 1
        assert_eq!(batches[3].len(), 1);
    }

    #[test]
    fn shuffle_is_deterministic() {
        let mut ds1 = ChessDataset::new();
        let mut ds2 = ChessDataset::new();
        ds1.extend(dummy_samples(20));
        ds2.extend(dummy_samples(20));
        ds1.shuffle(42);
        ds2.shuffle(42);
        for (a, b) in ds1.samples.iter().zip(ds2.samples.iter()) {
            assert_eq!(a.1, b.1);
        }
    }
}
