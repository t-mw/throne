use std::collections::HashMap;
use std::convert::TryInto;
use std::i32;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Atom {
    is_number: bool,
    v: u32,
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

        let idx = self.atom_to_str.len().try_into().unwrap();
        let atom = Atom {
            is_number: false,
            v: idx,
        };

        self.atom_to_str.push(text.to_string());
        self.str_to_atom.insert(text.to_string(), atom);

        atom
    }

    pub fn str_to_existing_atom(&self, text: &str) -> Option<Atom> {
        self.str_to_atom.get(text).cloned()
    }

    pub fn number_to_atom(n: i32) -> Atom {
        Atom {
            is_number: true,
            v: n as u32,
        }
    }

    pub fn atom_to_str(&self, atom: Atom) -> Option<&str> {
        if atom.is_number {
            None
        } else {
            Some(&self.atom_to_str[atom.v as usize])
        }
    }

    pub fn atom_to_number(atom: Atom) -> Option<i32> {
        if atom.is_number {
            Some(atom.v as i32)
        } else {
            None
        }
    }
}
