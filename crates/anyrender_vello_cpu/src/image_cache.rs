use std::collections::HashMap;
use std::mem::size_of;

use peniko::{ImageAlphaType, ImageData, ImageFormat};
use vello_cpu::ImageSource;

struct Entry {
    image: ImageSource,
    width: u32,
    height: u32,
    format: ImageFormat,
    alpha_type: ImageAlphaType,
    bytes: usize,
    used: u128,
}

/// A byte-bounded cache of converted CPU image sources.
///
/// Blob identities must describe immutable pixels. Replacing a Blob gives a
/// source a new version. Owners can release an obsolete version with `remove`.
/// Eviction releases the cache's reference; paints already using the image
/// retain their own reference until rendering completes.
pub struct ImageCache {
    entries: HashMap<u64, Entry>,
    limit: usize,
    bytes: usize,
    clock: u128,
    hits: u64,
    misses: u64,
}

impl Default for ImageCache {
    fn default() -> Self {
        Self::new(64 * 1024 * 1024)
    }
}

impl ImageCache {
    pub fn new(byte_limit: usize) -> Self {
        Self {
            entries: HashMap::new(),
            limit: byte_limit,
            bytes: 0,
            clock: 0,
            hits: 0,
            misses: 0,
        }
    }

    pub fn byte_limit(&self) -> usize {
        self.limit
    }

    pub fn retained_bytes(&self) -> usize {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    pub fn remove(&mut self, blob_id: u64) {
        if let Some(entry) = self.entries.remove(&blob_id) {
            self.bytes -= entry.bytes;
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }

    pub(crate) fn convert(&mut self, source: &ImageData) -> ImageSource {
        self.clock = self.clock.saturating_add(1);
        let id = source.data.id();
        if let Some(entry) = self.entries.get_mut(&id)
            && entry.width == source.width
            && entry.height == source.height
            && entry.format == source.format
            && entry.alpha_type == source.alpha_type
        {
            entry.used = self.clock;
            self.hits = self.hits.saturating_add(1);
            return entry.image.clone();
        }

        self.remove(id);
        self.misses = self.misses.saturating_add(1);
        let image = ImageSource::from_peniko_image_data(source);
        // Conversion produces an RGBA8 pixmap. Include the entry and its key
        // in the budget as well as the pixel allocation.
        let bytes = (source.width as usize)
            .checked_mul(source.height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .and_then(|pixels| pixels.checked_add(size_of::<Entry>() + size_of::<u64>()));
        let Some(bytes) = bytes.filter(|bytes| *bytes <= self.limit) else {
            return image;
        };

        while self.bytes > self.limit - bytes {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.used)
                .map(|(&id, _)| id);
            let Some(oldest) = oldest else {
                break;
            };
            self.remove(oldest);
        }
        self.bytes += bytes;
        self.entries.insert(
            id,
            Entry {
                image: image.clone(),
                width: source.width,
                height: source.height,
                format: source.format,
                alpha_type: source.alpha_type,
                bytes,
                used: self.clock,
            },
        );
        image
    }
}
