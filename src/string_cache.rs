use std::collections::HashMap;
use std::i32;

const MAX_STRING_IDX: AtomIdx = i32::MAX - MAX_NUMBER * 2 - 1;
const MAX_NUMBER: i32 = 99999;

pub type AtomIdx = i32;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Atom {
    pub idx: AtomIdx,
}

#[derive(Clone, Debug, Default)]
pub struct StringCache {
    atom_to_str: Vec<String>,
    str_to_atom: HashMap<String, Atom>,
    pub wildcard_counter: i32,
}

impl StringCache {
    pub fn new() -> StringCache {
        Default::default()
    }

    pub fn str_to_atom(&mut self, text: &str) -> Atom {
        use std::str::FromStr;

        if let Some(n) = i32::from_str(text).ok() {
            return StringCache::number_to_atom(n);
        }

        if let Some(atom) = self.str_to_existing_atom(text) {
            return atom;
        }

        let idx = self.atom_to_str.len() as AtomIdx;
        if idx > MAX_STRING_IDX {
            panic!("String cache full");
        }

        let atom = Atom { idx };

        self.atom_to_str.push(text.to_string());
        self.str_to_atom.insert(text.to_string(), atom);

        atom
    }

    pub fn str_to_existing_atom(&self, text: &str) -> Option<Atom> {
        self.str_to_atom.get(text).cloned()
    }

    pub fn number_to_atom(n: i32) -> Atom {
        if n.abs() > MAX_NUMBER {
            panic!("{} is large than the maximum of {}", n.abs(), MAX_NUMBER);
        }

        return Atom {
            idx: (n + MAX_NUMBER + 1) + MAX_STRING_IDX,
        };
    }

    pub fn atom_to_str(&self, atom: Atom) -> Option<&str> {
        if atom.idx <= MAX_STRING_IDX {
            Some(&self.atom_to_str[atom.idx as usize])
        } else {
            None
        }
    }

    pub fn atom_to_number(atom: Atom) -> Option<i32> {
        if atom.idx <= MAX_STRING_IDX {
            None
        } else {
            Some((atom.idx - MAX_STRING_IDX) - MAX_NUMBER - 1)
        }
    }
}
