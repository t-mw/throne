use std::collections::HashMap;
use std::convert::TryInto;
use std::i32;

/// References a string or integer stored in a [StringCache].
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Atom {
    is_integer: bool,
    v: u32,
}

/// Stores the mappings between a set of [Atoms](Atom) and the primitives that they reference.
#[derive(Clone, Debug)]
pub struct StringCache {
    atom_to_str: Vec<String>,
    str_to_atom: HashMap<String, Atom>,
    pub(crate) wildcard_counter: i32,
}

impl StringCache {
    pub(crate) fn new() -> StringCache {
        StringCache {
            atom_to_str: vec![],
            str_to_atom: HashMap::new(),
            wildcard_counter: 0,
        }
    }

    /// Returns an [Atom] referencing the provided `string`, defining a new [Atom] if necessary.
    ///
    /// The string referenced by the returned [Atom] can be retrieved using [Self::atom_to_str()],
    /// unless the string could be converted to an integer in which case [Self::atom_to_integer()]
    /// must be used.
    pub fn str_to_atom(&mut self, string: &str) -> Atom {
        use std::str::FromStr;

        if let Some(n) = i32::from_str(string).ok() {
            return StringCache::integer_to_atom(n);
        }

        if let Some(atom) = self.str_to_existing_atom(string) {
            return atom;
        }

        let idx = self.atom_to_str.len().try_into().unwrap();
        let atom = Atom {
            is_integer: false,
            v: idx,
        };

        self.atom_to_str.push(string.to_string());
        self.str_to_atom.insert(string.to_string(), atom);

        atom
    }

    /// Returns an [Atom] referencing the provided `string`, only if an [Atom] was previously defined for the string.
    pub fn str_to_existing_atom(&self, string: &str) -> Option<Atom> {
        self.str_to_atom.get(string).cloned()
    }

    /// Returns an [Atom] referencing the provided `integer`.
    pub fn integer_to_atom(integer: i32) -> Atom {
        Atom {
            is_integer: true,
            v: integer as u32,
        }
    }

    /// Returns the string referenced by the provided `atom`.
    pub fn atom_to_str(&self, atom: Atom) -> Option<&str> {
        if atom.is_integer {
            None
        } else {
            Some(&self.atom_to_str[atom.v as usize])
        }
    }

    /// Returns the integer referenced by the provided `atom`.
    pub fn atom_to_integer(atom: Atom) -> Option<i32> {
        if atom.is_integer {
            Some(atom.v as i32)
        } else {
            None
        }
    }
}
