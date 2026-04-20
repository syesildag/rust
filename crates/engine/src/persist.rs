//! `Persist` trait — unified save/load interface for types that can be written
//! to and read from binary files.
//!
//! Implementors provide only `write_to` and `read_from`; the default
//! `save_to` / `load_from` methods handle all `BufWriter`/`BufReader`/`File`
//! boilerplate.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use tensor::optim::Adam;
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

#[allow(clippy::cast_possible_truncation)]
impl Persist for Adam {
    fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        let (t, m, v) = self.state();
        w.write_all(&(t as u64).to_le_bytes())?;
        w.write_all(&(m.len() as u64).to_le_bytes())?;
        for (mi, vi) in m.iter().zip(v.iter()) {
            w.write_all(&(mi.len() as u64).to_le_bytes())?;
            for &x in mi {
                w.write_all(&x.to_le_bytes())?;
            }
            for &x in vi {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        Ok(())
    }

    fn read_from<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let mut buf8 = [0u8; 8];
        let mut buf4 = [0u8; 4];

        r.read_exact(&mut buf8)?;
        let t = u64::from_le_bytes(buf8) as usize;

        r.read_exact(&mut buf8)?;
        let num_groups = u64::from_le_bytes(buf8) as usize;

        let mut m = Vec::with_capacity(num_groups);
        let mut v = Vec::with_capacity(num_groups);
        for _ in 0..num_groups {
            r.read_exact(&mut buf8)?;
            let len = u64::from_le_bytes(buf8) as usize;
            let mut mi = vec![0f32; len];
            let mut vi = vec![0f32; len];
            for x in &mut mi {
                r.read_exact(&mut buf4)?;
                *x = f32::from_le_bytes(buf4);
            }
            for x in &mut vi {
                r.read_exact(&mut buf4)?;
                *x = f32::from_le_bytes(buf4);
            }
            m.push(mi);
            v.push(vi);
        }
        Ok(Adam::from_state(t, m, v))
    }
}
