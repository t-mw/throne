use rand::{rngs::SmallRng, seq::SliceRandom};

use crate::matching::phrase_equal;
use crate::string_cache::Atom;
use crate::token::*;

use std::ops::Range;

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct PhraseId {
    idx: usize,
    rev: usize,
}

#[derive(Clone, Debug)]
pub struct State {
    // indexes into token collection
    phrase_ranges: Vec<Range<usize>>,
    // collection of all tokens found in the state phrases
    tokens: Vec<Token>,

    removed_token_ranges: Vec<Range<usize>>,

    pub first_atoms: Vec<(usize, Atom)>,
    scratch_idx: Option<(usize, usize)>,

    // increments on mutation
    rev: usize,
}

impl State {
    pub fn new() -> State {
        State {
            phrase_ranges: vec![],
            tokens: vec![],
            removed_token_ranges: vec![],
            first_atoms: vec![],
            scratch_idx: None,
            rev: 0,
        }
    }

    pub fn remove(&mut self, id: PhraseId) {
        assert!(id.rev == self.rev);
        self.remove_idx(id.idx);
    }

    pub(crate) fn remove_idx(&mut self, idx: usize) {
        assert!(self.scratch_idx.is_none());

        let remove_range = self.phrase_ranges.swap_remove(idx);
        self.removed_token_ranges.push(remove_range);

        self.rev += 1;
    }

    pub fn remove_phrase(&mut self, phrase: &Phrase) {
        let remove_idx = self
            .phrase_ranges
            .iter()
            .position(|range| phrase_equal(&self.tokens[range.clone()], phrase, (0, 0), (0, 0)))
            .expect("remove_idx");

        self.remove_idx(remove_idx);
    }

    pub fn remove_pattern<const N: usize>(
        &mut self,
        pattern: [Option<Atom>; N],
        match_pattern_length: bool,
    ) {
        assert!(self.scratch_idx.is_none());

        let tokens = &mut self.tokens;
        let removed_token_ranges = &mut self.removed_token_ranges;
        let mut did_remove_tokens = false;

        self.phrase_ranges.retain(|range| {
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

            removed_token_ranges.push(range.clone());
            did_remove_tokens = true;

            false
        });

        if did_remove_tokens {
            self.rev += 1;
        }
    }

    pub fn clear_removed_tokens(&mut self) {
        self.removed_token_ranges
            .sort_by_key(|range| std::cmp::Reverse(range.start));
        for remove_range in self.removed_token_ranges.drain(..) {
            let remove_len = remove_range.end - remove_range.start;
            self.tokens.drain(remove_range.start..remove_range.end);
            for range in self.phrase_ranges.iter_mut() {
                if range.start >= remove_range.end {
                    range.start -= remove_len;
                    range.end -= remove_len;
                }
            }
        }
    }

    pub fn update_first_atoms(&mut self) {
        self.first_atoms = extract_first_atoms_state(self);
    }

    pub fn shuffle(&mut self, rng: &mut SmallRng) {
        assert!(self.scratch_idx.is_none());
        self.phrase_ranges.shuffle(rng);

        self.rev += 1;
    }

    pub fn push(&mut self, mut phrase: Vec<Token>) -> PhraseId {
        let start = self.tokens.len();
        self.tokens.append(&mut phrase);
        let end = self.tokens.len();

        self.phrase_ranges.push(Range { start, end });
        self.rev += 1;

        let id = PhraseId {
            idx: self.phrase_ranges.len() - 1,
            rev: self.rev,
        };

        id
    }

    pub fn len(&self) -> usize {
        self.phrase_ranges.len()
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = PhraseId> + 'a {
        let rev = self.rev;
        self.phrase_ranges
            .iter()
            .enumerate()
            .map(move |(idx, _)| PhraseId { idx, rev })
    }

    pub fn get(&self, id: PhraseId) -> &Phrase {
        assert!(id.rev == self.rev);
        &self.tokens[self.phrase_ranges[id.idx].clone()]
    }

    pub fn get_all(&self) -> Vec<Vec<Token>> {
        self.phrase_ranges
            .iter()
            .map(|range| self.tokens[range.clone()].to_vec())
            .collect::<Vec<_>>()
    }

    pub fn from_phrases(phrases: &[Vec<Token>]) -> State {
        let mut state = State::new();
        for p in phrases {
            state.push(p.clone());
        }
        state.update_first_atoms();
        state
    }

    pub fn lock_scratch(&mut self) {
        self.scratch_idx = Some((self.phrase_ranges.len(), self.tokens.len()));
    }

    pub fn unlock_scratch(&mut self) {
        self.reset_scratch();
        self.scratch_idx = None;
    }

    pub fn reset_scratch(&mut self) {
        let (phrase_ranges_len, token_len) = self.scratch_idx.expect("scratch_idx");
        self.phrase_ranges.drain(phrase_ranges_len..);
        self.tokens.drain(token_len..);
    }
}

impl std::ops::Index<usize> for State {
    type Output = [Token];

    fn index(&self, i: usize) -> &Phrase {
        &self.tokens[self.phrase_ranges[i].clone()]
    }
}

impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.get_all())
    }
}

fn extract_first_atoms_state(state: &State) -> Vec<(usize, Atom)> {
    let mut atoms: Vec<(usize, Atom)> = state
        .iter()
        .enumerate()
        .map(|(s_i, phrase_id)| (s_i, state.get(phrase_id)[0].atom))
        .collect();

    atoms.sort_unstable_by(|a, b| a.1.cmp(&b.1));

    atoms
}
