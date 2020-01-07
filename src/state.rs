use rand::rngs::SmallRng;
use rand::Rng;

use crate::matching::{phrase_equal, FirstAtoms};
use crate::token::*;

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct PhraseId {
    idx: usize,
}

#[derive(Clone, Debug)]
pub struct State {
    phrases: Vec<PhraseId>,
    phrase_ranges: Vec<(usize, usize)>,
    tokens: Vec<Token>,
    pub first_atoms: Vec<(usize, FirstAtoms)>,
    scratch_idx: Option<(usize, usize)>,
}

impl State {
    pub fn new() -> State {
        State {
            phrases: vec![],
            phrase_ranges: vec![],
            tokens: vec![],
            first_atoms: vec![],
            scratch_idx: None,
        }
    }

    pub fn remove(&mut self, idx: usize) {
        assert!(self.scratch_idx.is_none());

        let remove_id = self.phrases.swap_remove(idx);
        let remove_range = self.phrase_ranges[remove_id.idx];
        let remove_len = remove_range.1 - remove_range.0;

        // after swap_remove, this is the id that will take the old one's place
        let replace_id = PhraseId {
            idx: self.phrase_ranges.len() - 1,
        };
        self.phrase_ranges.swap_remove(remove_id.idx);

        // update the references to the swapped id
        for id in self.phrases.iter_mut() {
            if *id == replace_id {
                *id = remove_id;
            }
        }

        self.tokens.drain(remove_range.0..remove_range.1);

        for range in self.phrase_ranges.iter_mut() {
            if range.0 >= remove_range.1 {
                range.0 -= remove_len;
                range.1 -= remove_len;
            }
        }
    }

    pub fn update_first_atoms(&mut self) {
        self.first_atoms = extract_first_atoms_state(self);
    }

    pub fn remove_phrase(&mut self, phrase: &Phrase) {
        let remove_idx = self
            .phrases
            .iter()
            .position(|v| phrase_equal(self.get(*v), phrase, (0, 0), (0, 0)))
            .expect("remove_idx");

        self.remove(remove_idx);
    }

    pub fn shuffle(&mut self, rng: &mut SmallRng) {
        assert!(self.scratch_idx.is_none());
        rng.shuffle(&mut self.phrases);
    }

    pub fn push(&mut self, mut phrase: Vec<Token>) -> PhraseId {
        let begin = self.tokens.len();
        self.tokens.append(&mut phrase);
        let end = self.tokens.len();

        let idx = self.phrase_ranges.len();
        self.phrase_ranges.push((begin, end));

        let id = PhraseId { idx };
        self.phrases.push(id);

        id
    }

    pub fn len(&self) -> usize {
        self.phrases.len()
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a PhraseId> {
        self.phrases.iter()
    }

    pub fn get(&self, id: PhraseId) -> &Phrase {
        let (begin, end) = self.phrase_ranges[id.idx];
        &self.tokens[begin..end]
    }

    pub fn get_all(&self) -> Vec<Vec<Token>> {
        self.phrases
            .iter()
            .map(|id| {
                let (begin, end) = self.phrase_ranges[id.idx];
                self.tokens[begin..end].to_vec()
            })
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
        assert!(self.phrases.len() == self.phrase_ranges.len());
        self.scratch_idx = Some((self.phrases.len(), self.tokens.len()));
    }

    pub fn unlock_scratch(&mut self) {
        self.reset_scratch();
        self.scratch_idx = None;
    }

    pub fn reset_scratch(&mut self) {
        let (phrase_len, token_len) = self.scratch_idx.expect("scratch_idx");
        self.phrases.drain(phrase_len..);
        self.phrase_ranges.drain(phrase_len..);
        self.tokens.drain(token_len..);
    }
}

impl std::ops::Index<usize> for State {
    type Output = [Token];

    fn index(&self, i: usize) -> &Phrase {
        let id = self.phrases[i];
        self.get(id)
    }
}

impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.get_all())
    }
}

fn extract_first_atoms_state(state: &State) -> Vec<(usize, FirstAtoms)> {
    let mut atoms: Vec<(usize, FirstAtoms)> = state
        .iter()
        .enumerate()
        .map(|(s_i, phrase_id)| {
            let s = state.get(*phrase_id);
            (
                s_i,
                FirstAtoms {
                    a0: s.get(0).map(|t| t.string),
                    a1: s.get(1).map(|t| t.string),
                    a2: s.get(2).map(|t| t.string),
                },
            )
        })
        .collect();

    atoms.sort_unstable_by(|a, b| a.1.cmp(&b.1));

    atoms
}
