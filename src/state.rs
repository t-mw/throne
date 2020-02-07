use rand::rngs::SmallRng;
use rand::Rng;

use crate::matching::phrase_equal;
use crate::string_cache::Atom;
use crate::token::*;

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct PhraseId {
    idx: usize,
    rev: usize,
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
struct InnerPhraseId {
    phrase_range_idx: usize,
}

#[derive(Clone, Debug)]
pub struct State {
    phrases: Vec<InnerPhraseId>,
    phrase_ranges: Vec<(usize, usize)>,
    tokens: Vec<Token>,
    pub first_atoms: Vec<(usize, Atom)>,
    scratch_idx: Option<(usize, usize)>,

    // increments on mutation
    rev: usize,
}

impl State {
    pub fn new() -> State {
        State {
            phrases: vec![],
            phrase_ranges: vec![],
            tokens: vec![],
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

        let remove_id = self.phrases.swap_remove(idx);
        let remove_range = self.phrase_ranges[remove_id.phrase_range_idx];
        let remove_len = remove_range.1 - remove_range.0;

        // after swap_remove, this is the id that will take the old one's place
        let swapped_id = InnerPhraseId {
            phrase_range_idx: self.phrase_ranges.len() - 1,
        };

        // we need to update phrases to point to the new phrase range index
        let replace_id = InnerPhraseId {
            phrase_range_idx: remove_id.phrase_range_idx,
        };

        self.phrase_ranges.swap_remove(remove_id.phrase_range_idx);

        for id in self.phrases.iter_mut() {
            if *id == swapped_id {
                *id = replace_id;
            }
        }

        self.tokens.drain(remove_range.0..remove_range.1);

        for range in self.phrase_ranges.iter_mut() {
            if range.0 >= remove_range.1 {
                range.0 -= remove_len;
                range.1 -= remove_len;
            }
        }

        self.rev += 1;
    }

    pub fn update_first_atoms(&mut self) {
        self.first_atoms = extract_first_atoms_state(self);
    }

    pub fn remove_phrase(&mut self, phrase: &Phrase) {
        let remove_idx = self
            .phrases
            .iter()
            .position(|v| phrase_equal(self.get_inner(*v), phrase, (0, 0), (0, 0)))
            .expect("remove_idx");

        self.remove_idx(remove_idx);
    }

    pub fn shuffle(&mut self, rng: &mut SmallRng) {
        assert!(self.scratch_idx.is_none());
        rng.shuffle(&mut self.phrases);

        self.rev += 1;
    }

    pub fn push(&mut self, mut phrase: Vec<Token>) -> PhraseId {
        let begin = self.tokens.len();
        self.tokens.append(&mut phrase);
        let end = self.tokens.len();

        let phrase_range_idx = self.phrase_ranges.len();
        self.phrase_ranges.push((begin, end));

        let inner_id = InnerPhraseId { phrase_range_idx };
        self.phrases.push(inner_id);

        self.rev += 1;

        let id = PhraseId {
            idx: self.phrases.len() - 1,
            rev: self.rev,
        };

        id
    }

    pub fn len(&self) -> usize {
        self.phrases.len()
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = PhraseId> + 'a {
        let rev = self.rev;
        self.phrases
            .iter()
            .enumerate()
            .map(move |(idx, _)| PhraseId { idx, rev })
    }

    pub fn get(&self, id: PhraseId) -> &Phrase {
        self.get_inner(self.phrases[id.idx])
    }

    fn get_inner(&self, inner_id: InnerPhraseId) -> &Phrase {
        let (begin, end) = self.phrase_ranges[inner_id.phrase_range_idx];
        &self.tokens[begin..end]
    }

    pub fn get_all(&self) -> Vec<Vec<Token>> {
        self.phrases
            .iter()
            .map(|id| {
                let (begin, end) = self.phrase_ranges[id.phrase_range_idx];
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
        self.get_inner(self.phrases[i])
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
