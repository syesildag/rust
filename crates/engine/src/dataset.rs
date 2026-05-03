//! `ChessDataset`: holds (Board, outcome) samples and provides shuffled mini-batches.

#![allow(clippy::cast_possible_truncation)]

use crate::fen_file::{parse_csv_file, parse_fen_file};
use crate::persist::Persist;
use crate::pgn::{parse_pgn, Sample};
use chess::fen;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
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
    /// Supports `.pgn`, `.fen`, `.epd`, and `.csv` files.  Directories are
    /// scanned for all files with those extensions.  Unreadable files are
    /// skipped with a warning on stderr.
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

    /// Derives the cache file path from a data file path.
    /// E.g. `"games.pgn"` → `"games.pgn.dscache"`
    #[must_use]
    pub fn cache_path(data: &Path) -> PathBuf {
        let mut p = data.to_owned();
        p.as_mut_os_string().push(".dscache");
        p
    }

    /// Loads from per-file binary caches if valid, otherwise parses and saves each cache.
    ///
    /// Each data file gets its own `.dscache` sidecar. The cache is valid when it
    /// exists and is newer than its corresponding source file.
    #[must_use]
    pub fn load_with_cache(paths: &[PathBuf]) -> Self {
        let mut ds = Self::new();
        for path in paths {
            if path.is_dir() {
                let entries = match fs::read_dir(path) {
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
                    ds.extend(Self::load_file_cached(p).samples);
                }
            } else {
                ds.extend(Self::load_file_cached(path).samples);
            }
        }
        ds
    }

    fn load_file_cached(path: &Path) -> Self {
        let cache = Self::cache_path(path);
        if cache_is_valid_single(&cache, path) {
            match Self::load_from(&cache) {
                Ok(ds) => {
                    info!(samples = ds.len(), path = %path.display(), "loaded file from cache");
                    return ds;
                }
                Err(e) => warn!(error = %e, "cache unreadable — re-parsing"),
            }
        }
        let mut ds = Self::new();
        ds.load_one(path);
        if let Err(e) = ds.save_to(&cache) {
            warn!(error = %e, "could not save dataset cache");
        } else {
            info!(samples = ds.len(), path = %path.display(), "dataset cache written");
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
                } else if is_csv_extension(path) {
                    parse_csv_file(&text)
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
/// `[u64 fen_len] [u8; fen_len] [f32 outcome] [u64 game_id]`
impl Persist for ChessDataset {
    fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        w.write_all(&(self.samples.len() as u64).to_le_bytes())?;
        for (board, outcome, game_id) in &self.samples {
            let fen_bytes = board.to_fen();
            let bytes = fen_bytes.as_bytes();
            w.write_all(&(bytes.len() as u64).to_le_bytes())?;
            w.write_all(bytes)?;
            w.write_all(&outcome.to_le_bytes())?;
            w.write_all(&game_id.to_le_bytes())?;
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
            let outcome = f32::from_le_bytes(buf4);
            r.read_exact(&mut buf8)?;
            let game_id = u64::from_le_bytes(buf8);
            samples.push((board, outcome, game_id));
        }

        Ok(Self { samples })
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Returns `true` if the cache file exists and is newer than its source file.
fn cache_is_valid_single(cache_path: &Path, source: &Path) -> bool {
    let Ok(cache_mtime) = fs::metadata(cache_path).and_then(|m| m.modified()) else {
        return false;
    };
    let Ok(src_mtime) = fs::metadata(source).and_then(|m| m.modified()) else {
        return false;
    };
    cache_mtime.duration_since(src_mtime).is_ok()
}

fn is_fen_extension(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("fen") || ext.eq_ignore_ascii_case("epd"))
}

fn is_csv_extension(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("csv"))
}

fn is_supported_extension(path: &Path) -> bool {
    path.extension().is_some_and(|ext| {
        ext.eq_ignore_ascii_case("pgn")
            || ext.eq_ignore_ascii_case("fen")
            || ext.eq_ignore_ascii_case("epd")
            || ext.eq_ignore_ascii_case("csv")
    })
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use chess::board::Board;

    fn dummy_samples(n: usize) -> Vec<Sample> {
        (0..n)
            .map(|_| (Board::starting_position(), 0.0f32, 0u64))
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
