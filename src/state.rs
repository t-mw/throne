use crate::matching::phrase_equal;
use crate::string_cache::Atom;
use crate::token::*;

use rand::{rngs::SmallRng, seq::SliceRandom};

use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::Hasher;
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

        let remove_range = self.storage.phrase_ranges.swap_remove(idx);
        self.storage.removed_phrase_ranges.push(remove_range);

        self.storage.rev += 1;
    }

    pub fn remove_phrase(&mut self, phrase: &Phrase) {
        let remove_idx = self
            .storage
            .phrase_ranges
            .iter()
            .position(|range| {
                phrase_equal(&self.storage.tokens[range.clone()], phrase, (0, 0), (0, 0))
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

        self.storage.phrase_ranges.retain(|range| {
            let phrase = &tokens[range.clone()];

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

            removed_phrase_ranges.push(range.clone());
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
            for range in self.storage.phrase_ranges.iter_mut() {
                if range.start >= remove_range.end {
                    range.start -= remove_len;
                    range.end -= remove_len;
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
    ) -> Option<Vec<usize>> {
        assert!(self.match_cache.storage_rev == self.storage.rev);
        self.match_cache.match_rule_input(input_phrase, self.len())
    }

    pub fn shuffle(&mut self, rng: &mut SmallRng) {
        assert!(self.scratch_state.is_none());
        self.storage.phrase_ranges.shuffle(rng);
        self.storage.rev += 1;
    }

    pub fn push(&mut self, mut phrase: Vec<Token>) -> PhraseId {
        let start = self.storage.tokens.len();
        self.storage.tokens.append(&mut phrase);
        let end = self.storage.tokens.len();

        self.storage.phrase_ranges.push(Range { start, end });
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
            .map(|range| self.storage.tokens[range.clone()].to_vec())
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
        &self.storage.tokens[self.storage.phrase_ranges[i].clone()]
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
    phrase_ranges: Vec<Range<usize>>,
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
        &self.tokens[self.phrase_ranges[id.idx].clone()]
    }
}

#[derive(Clone, Debug)]
struct MatchCache {
    first_atoms: Vec<(usize, Atom)>,
    by_length: HashMap<usize, Vec<usize>>,
    by_group_hash_position: HashMap<(u64, usize), Vec<usize>>,
    storage_rev: usize,
}

impl MatchCache {
    fn new() -> Self {
        MatchCache {
            first_atoms: vec![],
            by_length: HashMap::new(),
            by_group_hash_position: HashMap::new(),
            storage_rev: 0,
        }
    }

    fn clear(&mut self) {
        self.first_atoms.clear();
        self.by_length.clear();
        self.by_group_hash_position.clear();
    }

    fn update_storage(&mut self, storage: &Storage) {
        if self.storage_rev == storage.rev {
            return;
        }
        self.storage_rev = storage.rev;

        self.clear();
        self.first_atoms = extract_first_atoms_state(storage);

        for (s_i, phrase_id) in storage.iter().enumerate() {
            let phrase = storage.get(phrase_id);
            self.cache_state_phrase_matches(s_i, phrase);
        }
    }

    fn cache_state_phrase_matches(&mut self, state_index: usize, phrase: &Phrase) {
        let group_count = phrase.groups().count();
        self.by_length
            .entry(group_count)
            .or_default()
            .push(state_index);
        for (g_i, g) in phrase.groups().enumerate() {
            let mut hasher = DefaultHasher::new();
            for token in g {
                token.hash_for_matching(&mut hasher);
            }
            let group_hash = hasher.finish();
            self.by_group_hash_position
                .entry((group_hash, g_i))
                .or_default()
                .push(state_index);
        }
    }

    fn match_rule_input(&self, input_phrase: &Phrase, storage_length: usize) -> Option<Vec<usize>> {
        if let Some(rule_first_atom) = extract_first_atom_rule_input(input_phrase) {
            let start_idx = if let Ok(idx) = self
                .first_atoms
                .binary_search_by(|probe| probe.1.cmp(&rule_first_atom))
            {
                // binary search won't always find the first match,
                // so search backwards until we find it
                self.first_atoms
                    .iter()
                    .enumerate()
                    .rev()
                    .skip(self.first_atoms.len() - 1 - idx)
                    .take_while(|(_, a)| a.1 == rule_first_atom)
                    .last()
                    .expect("start_idx")
                    .0
            } else {
                return None;
            };

            return Some(
                self.first_atoms
                    .iter()
                    .skip(start_idx)
                    .take_while(|(_, a)| *a == rule_first_atom)
                    .map(|(s_i, _)| *s_i)
                    .collect(),
            );
        }

        let mut sets = vec![self.by_length.get(&input_phrase.groups().count())?];
        for (g_i, g) in input_phrase.groups().enumerate() {
            if g.len() > 1 || !is_concrete_pred(g) {
                continue;
            }
            let mut hasher = DefaultHasher::new();
            for token in g {
                token.hash_for_matching(&mut hasher);
            }
            let group_hash = hasher.finish();
            sets.push(self.by_group_hash_position.get(&(group_hash, g_i))?);
        }

        // check rarest match first
        sets.sort_by_key(|s| s.len());

        let mut set_match_counts = vec![0; storage_length];
        for set in &sets {
            for s_i in *set {
                set_match_counts[*s_i] += 1;
            }
        }

        let matches: Vec<_> = set_match_counts
            .drain(..)
            .enumerate()
            .filter(|(_, count)| *count == sets.len())
            .map(|(s_i, _)| s_i)
            .collect();
        if matches.len() == 0 {
            None
        } else {
            Some(matches)
        }
    }
}

fn extract_first_atoms_state(storage: &Storage) -> Vec<(usize, Atom)> {
    let mut atoms: Vec<(usize, Atom)> = storage
        .iter()
        .enumerate()
        .map(|(s_i, phrase_id)| (s_i, storage.get(phrase_id)[0].atom))
        .collect();
    atoms.sort_unstable_by(|a, b| a.1.cmp(&b.1));
    atoms
}

fn extract_first_atom_rule_input(phrase: &Phrase) -> Option<Atom> {
    let first_group_is_single_token = phrase[0].open_depth == 1;
    if first_group_is_single_token && is_concrete_pred(phrase) {
        phrase.get(0).filter(|t| !is_var_token(t)).map(|t| t.atom)
    } else {
        None
    }
}
