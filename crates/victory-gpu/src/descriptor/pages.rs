//! Title descriptor indices on a host view heap smaller than the title's.
//!
//! The host caps shader-visible view heaps (NVIDIA at 1,000,000
//! descriptors), and Grand Theft Auto VI's main heap is larger. A title
//! uses few parts of its heap, so the title's indices are split into pages
//! and a page gets a host page the first time a view is written in it.
//! Inside a page the order stays, so a table is a run of host slots as long
//! as it stays in its page (or its pages got neighbouring host pages).

/// Descriptors per page.
pub(crate) const PAGE: u32 = 4096;

/// Where each title page lies on the host.
pub(crate) struct PageMap {
    /// The host page of each title page; [`NONE`] until one is given.
    /// Empty when the host heap holds the title's (indices stay).
    host: Vec<u32>,
    /// Which host pages are given.
    taken: Vec<bool>,
    /// Host slots this map hands out.
    slots: u32,
}

const NONE: u32 = u32::MAX;

/// No host page is left for a title page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Exhausted;

impl PageMap {
    /// A map for a title heap of `count` descriptors on at most `host`
    /// host slots.
    pub(crate) fn new(count: u32, host: u32) -> Self {
        if count <= host {
            return PageMap {
                host: Vec::new(),
                taken: Vec::new(),
                slots: count,
            };
        }
        let host_pages = host / PAGE;
        PageMap {
            host: vec![NONE; count.div_ceil(PAGE) as usize],
            taken: vec![false; host_pages as usize],
            slots: host_pages * PAGE,
        }
    }

    /// Host slots in use for title descriptors (`0..slots`).
    pub(crate) fn slots(&self) -> u32 {
        self.slots
    }

    /// The host slot of title descriptor `index`, if its page has one.
    pub(crate) fn find(&self, index: u32) -> Option<u32> {
        if self.host.is_empty() {
            return (index < self.slots).then_some(index);
        }
        let page = *self.host.get((index / PAGE) as usize)?;
        (page != NONE).then(|| page * PAGE + index % PAGE)
    }

    /// Whether the `n` title descriptors from `index` are a run of host
    /// slots: their pages have host pages, each after the one before.
    pub(crate) fn run(&self, index: u32, n: u32) -> bool {
        let Some(first) = self.find(index) else {
            return n == 0;
        };
        let Some(last) = n.checked_sub(1).map(|k| index.saturating_add(k)) else {
            return true;
        };
        (index / PAGE + 1..=last / PAGE)
            .all(|page| self.find(page * PAGE) == Some(first + (page * PAGE - index)))
            && self.find(last).is_some()
    }

    /// The host slot of title descriptor `index`, giving its page a host
    /// page if it has none.
    pub(crate) fn map(&mut self, index: u32) -> Result<u32, Exhausted> {
        if let Some(slot) = self.find(index) {
            return Ok(slot);
        }
        let page = (index / PAGE) as usize;
        if page >= self.host.len() {
            return Err(Exhausted);
        }
        // Its own place, so that shaders indexing the heap with an index
        // the title computed still read it; else the host page after the
        // previous title page's, so tables across the two stay a run; else
        // the first free one.
        let free = |p: usize| self.taken.get(p).is_some_and(|t| !t);
        let after = page
            .checked_sub(1)
            .map(|p| self.host[p])
            .filter(|&h| h != NONE)
            .map(|h| h as usize + 1);
        let free = if free(page) {
            page
        } else if let Some(after) = after.filter(|&h| free(h)) {
            after
        } else {
            self.taken.iter().position(|t| !t).ok_or(Exhausted)?
        };
        self.taken[free] = true;
        self.host[page] = free as u32;
        Ok(free as u32 * PAGE + index % PAGE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Grand Theft Auto VI's main view heap, and NVIDIA's limit for
    /// shader-visible view heaps less the dynamic slots.
    const TITLE: u32 = 1_245_184;
    const HOST: u32 = 1_000_000 - (1 << 16);

    #[test]
    fn a_heap_that_fits_keeps_title_indices() {
        let mut pages = PageMap::new(TITLE, TITLE);
        assert_eq!(pages.slots(), TITLE);
        assert_eq!(pages.map(TITLE - 1), Ok(TITLE - 1));
        assert_eq!(pages.find(12_345), Some(12_345));
    }

    #[test]
    fn writes_in_high_pages_land_inside_a_capped_heap() {
        let mut pages = PageMap::new(TITLE, HOST);
        assert!(pages.slots() <= HOST);
        assert_eq!(pages.slots() % PAGE, 0);
        for index in [TITLE - 1, TITLE - PAGE, 1_000_000, HOST] {
            let host = pages.map(index).unwrap();
            assert!(host < pages.slots(), "{index} -> {host}");
            // Mapped once: the same slot every time.
            assert_eq!(pages.find(index), Some(host));
            assert_eq!(pages.map(index), Ok(host));
        }
    }

    /// Shaders that index the heap with an index the title computed read
    /// the right descriptor in pages that kept their place.
    #[test]
    fn low_pages_keep_their_index_while_their_host_page_is_free() {
        let mut pages = PageMap::new(TITLE, HOST);
        assert_eq!(pages.map(5 * PAGE + 7), Ok(5 * PAGE + 7));
        // A high page takes a free host page, not one a low page keeps.
        let high = pages.map(TITLE - 1).unwrap();
        assert_ne!(high / PAGE, 5);
        // A low page whose place a high page took goes elsewhere.
        let taken = high / PAGE * PAGE;
        let moved = pages.map(taken).unwrap();
        assert_ne!(moved, taken);
        assert!(moved < pages.slots());
    }

    #[test]
    fn running_out_of_host_pages_is_an_error() {
        let mut pages = PageMap::new(3 * PAGE, 2 * PAGE);
        assert_eq!(pages.map(0), Ok(0));
        assert_eq!(pages.map(2 * PAGE), Ok(PAGE));
        assert_eq!(pages.map(PAGE + 1), Err(Exhausted));
        assert_eq!(pages.find(PAGE + 1), None);
        // Indices past the title heap have no page either.
        assert_eq!(pages.map(3 * PAGE), Err(Exhausted));
        // What was mapped stays.
        assert_eq!(pages.map(2 * PAGE + 3), Ok(PAGE + 3));
    }

    /// A table is read as a run of host slots from its first.
    #[test]
    fn tables_are_runs_only_where_their_pages_are() {
        // Six title pages on four host pages.
        let mut pages = PageMap::new(6 * PAGE, 4 * PAGE);
        // Title pages 4 and 5 have no place of their own: 4 takes host page
        // 0, and 5 the one after it.
        assert_eq!(pages.map(4 * PAGE), Ok(0));
        assert_eq!(pages.map(5 * PAGE), Ok(PAGE));
        assert!(pages.run(5 * PAGE - 2, 4));
        assert!(pages.run(4 * PAGE + 1, 8));
        // Title page 2 keeps its place; page 3 has no host page yet.
        assert_eq!(pages.map(2 * PAGE), Ok(2 * PAGE));
        assert!(!pages.run(3 * PAGE - 2, 4));
        assert!(!pages.run(3 * PAGE, 1));
        // Page 3 keeps its place too, after page 2's: a run again.
        assert_eq!(pages.map(3 * PAGE), Ok(3 * PAGE));
        assert!(pages.run(3 * PAGE - 2, 4));
        // Page 3 into page 4 (host page 0): not a run.
        assert!(!pages.run(4 * PAGE - 1, 2));
        // Past the heap, and an empty table.
        assert!(!pages.run(6 * PAGE - 1, 2));
        assert!(pages.run(0, 0));
    }
}
