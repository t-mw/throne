#[macro_use]
extern crate lazy_static;
extern crate pest;
#[macro_use]
extern crate pest_derive;
extern crate rand;
extern crate regex;

mod ceptre;
mod ffi;
mod matching;
mod parser;
mod rule;
mod state;
mod string_cache;
mod token;

pub use crate::ceptre::{update, Context, ContextBuilder, Core, PhraseString};
pub use crate::ffi::*;
pub use crate::rule::Rule;
pub use crate::state::State;
pub use crate::string_cache::{Atom, StringCache};
pub use crate::token::{build_phrase, tokenize, Phrase, PhraseGroup, Token, VecPhrase};
