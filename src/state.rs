use crate::matching::phrase_equal;
use crate::string_cache::Atom;
use crate::token::*;

use rand::{rngs::SmallRng, seq::SliceRandom};

use std::ops::Range;

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct PhraseId {
    idx: usize,
    rev: usize,
}

#[derive(Clone, Debug)]
pub struct State {
    storage: Storage,
    match_cache: MatchCache,
    scratch_state: Option<ScratchState>,
}

impl State {
    pub fn new() -> State {
        State {
            storage: Storage::new(),
            match_cache: MatchCache::new(),
            scratch_state: None,
        }
    }

    pub fn remove(&mut self, id: PhraseId) {
        assert!(id.rev == self.storage.rev);
        self.remove_idx(id.idx);
    }

    pub(crate) fn remove_idx(&mut self, idx: usize) {
        assert!(!self.is_locked());

        let remove_phrase = self.storage.phrase_ranges.swap_remove(idx);
        self.storage
            .removed_phrase_ranges
            .push(remove_phrase.token_range);

        self.storage.rev += 1;
    }

    pub fn remove_phrase(&mut self, phrase: &Phrase) {
        let remove_idx = self
            .storage
            .phrase_ranges
            .iter()
            .position(|PhraseMetadata { token_range, .. }| {
                phrase_equal(
                    &self.storage.tokens[token_range.clone()],
                    phrase,
                    (0, 0),
                    (0, 0),
                )
            })
            .expect("remove_idx");

        self.remove_idx(remove_idx);
    }

    pub fn remove_pattern<const N: usize>(
        &mut self,
        pattern: [Option<Atom>; N],
        match_pattern_length: bool,
    ) {
        assert!(!self.is_locked());

        let tokens = &mut self.storage.tokens;
        let removed_phrase_ranges = &mut self.storage.removed_phrase_ranges;
        let mut did_remove_tokens = false;

        self.storage
            .phrase_ranges
            .retain(|PhraseMetadata { token_range, .. }| {
                let phrase = &tokens[token_range.clone()];

                if phrase.len() < N || (match_pattern_length && phrase.len() != N) {
                    return true;
                }

                for (i, atom) in pattern.iter().enumerate() {
                    if let Some(atom) = atom {
                        if phrase[i].atom != *atom {
                            return true;
                        }
                    }
                }

                removed_phrase_ranges.push(token_range.clone());
                did_remove_tokens = true;

                false
            });

        if did_remove_tokens {
            self.storage.rev += 1;
        }
    }

    pub fn clear_removed_tokens(&mut self) {
        self.storage
            .removed_phrase_ranges
            .sort_by_key(|range| std::cmp::Reverse(range.start));
        for remove_range in self.storage.removed_phrase_ranges.drain(..) {
            let remove_len = remove_range.end - remove_range.start;
            self.storage
                .tokens
                .drain(remove_range.start..remove_range.end);
            for PhraseMetadata { token_range, .. } in self.storage.phrase_ranges.iter_mut() {
                if token_range.start >= remove_range.end {
                    token_range.start -= remove_len;
                    token_range.end -= remove_len;
                }
            }
        }
    }

    pub fn update_cache(&mut self) {
        self.match_cache.update_storage(&self.storage);
    }

    pub fn match_cached_state_indices_for_rule_input(
        &self,
        input_phrase: &Phrase,
        input_phrase_group_count: usize,
    ) -> &[usize] {
        assert!(self.match_cache.storage_rev == self.storage.rev);
        debug_assert_eq!(input_phrase.groups().count(), input_phrase_group_count);
        self.match_cache
            .match_rule_input(input_phrase, input_phrase_group_count)
    }

    pub fn shuffle(&mut self, rng: &mut SmallRng) {
        assert!(self.scratch_state.is_none());
        self.storage.phrase_ranges.shuffle(rng);
        self.storage.rev += 1;
    }

    pub fn push(&mut self, phrase: Vec<Token>) -> PhraseId {
        let group_count = phrase.groups().count();
        self.push_with_metadata(phrase, group_count)
    }

    pub(crate) fn push_with_metadata(
        &mut self,
        mut phrase: Vec<Token>,
        group_count: usize,
    ) -> PhraseId {
        let first_group_is_single_token = phrase[0].open_depth == 1;
        let first_atom = if first_group_is_single_token && is_concrete_pred(&phrase) {
            Some(phrase[0].atom)
        } else {
            None
        };

        let start = self.storage.tokens.len();
        self.storage.tokens.append(&mut phrase);
        let end = self.storage.tokens.len();

        self.storage.phrase_ranges.push(PhraseMetadata {
            token_range: Range { start, end },
            first_atom,
            group_count,
        });
        self.storage.rev += 1;

        let id = PhraseId {
            idx: self.storage.phrase_ranges.len() - 1,
            rev: self.storage.rev,
        };

        id
    }

    pub fn len(&self) -> usize {
        self.storage.phrase_ranges.len()
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = PhraseId> + 'a {
        self.storage.iter()
    }

    pub fn get(&self, id: PhraseId) -> &Phrase {
        self.storage.get(id)
    }

    pub fn get_all(&self) -> Vec<Vec<Token>> {
        self.storage
            .phrase_ranges
            .iter()
            .map(|PhraseMetadata { token_range, .. }| {
                self.storage.tokens[token_range.clone()].to_vec()
            })
            .collect::<Vec<_>>()
    }

    pub fn from_phrases(phrases: &[Vec<Token>]) -> State {
        let mut state = State::new();
        for p in phrases {
            state.push(p.clone());
        }
        state.update_cache();
        state
    }

    pub fn lock_scratch(&mut self) {
        self.scratch_state = Some(ScratchState {
            storage_phrase_ranges_len: self.storage.phrase_ranges.len(),
            storage_tokens_len: self.storage.tokens.len(),
            storage_rev: self.storage.rev,
        });
    }

    pub fn unlock_scratch(&mut self) {
        self.reset_scratch();
        self.scratch_state = None;
    }

    pub fn reset_scratch(&mut self) {
        let ScratchState {
            storage_phrase_ranges_len,
            storage_tokens_len,
            storage_rev,
            ..
        } = self.scratch_state.as_ref().expect("scratch_state");
        self.storage
            .phrase_ranges
            .drain(storage_phrase_ranges_len..);
        self.storage.tokens.drain(storage_tokens_len..);
        self.storage.rev = *storage_rev;
    }

    fn is_locked(&self) -> bool {
        self.scratch_state.is_some()
    }
}

impl std::ops::Index<usize> for State {
    type Output = [Token];

    fn index(&self, i: usize) -> &Phrase {
        self.storage.get_by_metadata(&self.storage.phrase_ranges[i])
    }
}

impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.get_all())
    }
}

#[derive(Clone, Debug)]
struct ScratchState {
    storage_phrase_ranges_len: usize,
    storage_tokens_len: usize,
    storage_rev: usize,
}

#[derive(Clone, Debug)]
struct Storage {
    // indexes into token collection
    phrase_ranges: Vec<PhraseMetadata>,
    removed_phrase_ranges: Vec<Range<usize>>,

    // collection of all tokens found in the state phrases
    tokens: Vec<Token>,

    // increments on mutation
    rev: usize,
}

impl Storage {
    fn new() -> Self {
        Storage {
            phrase_ranges: vec![],
            removed_phrase_ranges: vec![],
            tokens: vec![],
            rev: 0,
        }
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = PhraseId> + 'a {
        let rev = self.rev;
        self.phrase_ranges
            .iter()
            .enumerate()
            .map(move |(idx, _)| PhraseId { idx, rev })
    }

    fn get(&self, id: PhraseId) -> &Phrase {
        assert!(id.rev == self.rev);
        self.get_by_metadata(&self.phrase_ranges[id.idx])
    }

    fn get_by_metadata(&self, metadata: &PhraseMetadata) -> &Phrase {
        &self.tokens[metadata.token_range.clone()]
    }
}

#[derive(Clone, Debug)]
struct PhraseMetadata {
    token_range: Range<usize>,
    first_atom: Option<Atom>,
    group_count: usize,
}

#[derive(Clone, Debug)]
struct MatchCache {
    first_atom_pairs: Vec<(Atom, usize)>,
    first_atom_indices: Vec<usize>,
    state_indices_by_length: Vec<Vec<usize>>,
    storage_rev: usize,
}

impl MatchCache {
    fn new() -> Self {
        MatchCache {
            first_atom_pairs: vec![],
            first_atom_indices: vec![],
            state_indices_by_length: vec![],
            storage_rev: 0,
        }
    }

    fn clear(&mut self) {
        self.first_atom_pairs.clear();
        self.first_atom_indices.clear();
        self.state_indices_by_length.clear();
    }

    fn update_storage(&mut self, storage: &Storage) {
        if self.storage_rev == storage.rev {
            return;
        }
        self.storage_rev = storage.rev;

        self.clear();
        for (s_i, phrase_metadata) in storage.phrase_ranges.iter().enumerate() {
            if let Some(first_atom) = phrase_metadata.first_atom {
                self.first_atom_pairs.push((first_atom, s_i));
            }
            if self.state_indices_by_length.len() < phrase_metadata.group_count + 1 {
                self.state_indices_by_length
                    .resize(phrase_metadata.group_count + 1, vec![]);
            }
            self.state_indices_by_length[phrase_metadata.group_count].push(s_i);
        }
        self.first_atom_pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        for (_, s_i) in &self.first_atom_pairs {
            self.first_atom_indices.push(*s_i);
        }
    }

    fn match_rule_input(&self, input_phrase: &Phrase, input_phrase_group_count: usize) -> &[usize] {
        let first_group_is_single_token = input_phrase[0].open_depth == 1;
        if first_group_is_single_token && is_concrete_pred(input_phrase) {
            let input_first_atom = input_phrase[0].atom;
            if let Ok(idx) = self
                .first_atom_pairs
                .binary_search_by(|(atom, _)| atom.cmp(&input_first_atom))
            {
                // binary search won't always find the first match,
                // so search backwards until we find it
                let start_idx = self
                    .first_atom_pairs
                    .iter()
                    .enumerate()
                    .rev()
                    .skip(self.first_atom_pairs.len() - 1 - idx)
                    .take_while(|(_, (atom, _))| *atom == input_first_atom)
                    .last()
                    .expect("start idx")
                    .0;
                let end_idx = self
                    .first_atom_pairs
                    .iter()
                    .enumerate()
                    .skip(idx)
                    .take_while(|(_, (atom, _))| *atom == input_first_atom)
                    .last()
                    .expect("end idx")
                    .0;
                return &self.first_atom_indices[start_idx..end_idx + 1];
            } else {
                return &[];
            };
        }

        if let Some(v) = &self.state_indices_by_length.get(input_phrase_group_count) {
            v
        } else {
            &[]
        }
    }
}
