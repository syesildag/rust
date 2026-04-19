//! `ChessDataset`: holds (Board, outcome) samples and provides shuffled mini-batches.

use crate::fen_file::parse_fen_file;
use crate::pgn::{parse_pgn, Sample};
use std::path::Path;

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
                        eprintln!("Warning: cannot read dir {}: {e}", path.display());
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
                println!(
                    "  Loaded {} positions from {}",
                    self.samples.len() - before,
                    path.display()
                );
            }
            Err(e) => eprintln!("Warning: cannot read {}: {e}", path.display()),
        }
    }
}

impl Default for ChessDataset {
    fn default() -> Self {
        Self::new()
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn is_fen_extension(path: &Path) -> bool {
    path.extension().is_some_and(|ext| {
        ext.eq_ignore_ascii_case("fen") || ext.eq_ignore_ascii_case("epd")
    })
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
