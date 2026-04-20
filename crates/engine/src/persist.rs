//! `Persist` trait — unified save/load interface for types that can be written
//! to and read from binary files.
//!
//! Implementors provide only `write_to` and `read_from`; the default
//! `save_to` / `load_from` methods handle all `BufWriter`/`BufReader`/`File`
//! boilerplate.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use tracing::debug;

pub trait Persist: Sized {
    /// Serialises `self` into `w`.
    ///
    /// # Errors
    /// Returns an I/O error if writing fails.
    fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()>;

    /// Deserialises an instance from `r`.
    ///
    /// # Errors
    /// Returns an I/O error if reading fails or the data is invalid.
    fn read_from<R: Read>(r: &mut R) -> std::io::Result<Self>;

    /// Writes to `path`, wrapping the file in a `BufWriter`.
    ///
    /// # Errors
    /// Returns an I/O error if the file cannot be created or written.
    fn save_to(&self, path: &Path) -> std::io::Result<()> {
        debug!(path = %path.display(), "saving");
        let mut w = BufWriter::new(File::create(path)?);
        self.write_to(&mut w)?;
        w.flush()
    }

    /// Reads from `path`, wrapping the file in a `BufReader`.
    ///
    /// # Errors
    /// Returns an I/O error if the file cannot be opened or the data is invalid.
    fn load_from(path: &Path) -> std::io::Result<Self> {
        let mut r = BufReader::new(File::open(path)?);
        Self::read_from(&mut r)
    }
}
