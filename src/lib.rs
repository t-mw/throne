#[macro_use]
extern crate lazy_static;
extern crate pest;
#[macro_use]
extern crate pest_derive;
extern crate rand;
extern crate regex;

mod ceptre;
mod ffi;

pub use ceptre::{
    update, Atom, Context, Core, Phrase, PhraseGroup, PhraseString, Rule, State, StringCache, Token,
};
pub use ffi::*;
