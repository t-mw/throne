#[macro_use]
extern crate lazy_static;
extern crate pest;
#[macro_use]
extern crate pest_derive;
extern crate rand;
extern crate regex;

mod context;
mod core;
#[cfg(not(target_arch = "wasm32"))]
mod ffi;
mod matching;
mod parser;
mod rule;
mod state;
mod string_cache;
#[cfg(test)]
mod tests;
mod token;
mod update;
#[cfg(target_arch = "wasm32")]
mod wasm;

pub use crate::context::{Context, ContextBuilder};
pub use crate::core::Core;
#[cfg(not(target_arch = "wasm32"))]
pub use crate::ffi::*;
pub use crate::rule::Rule;
pub use crate::state::State;
pub use crate::string_cache::{Atom, StringCache};
pub use crate::token::{
    phrase_to_string, tokenize, Phrase, PhraseGroup, PhraseString, Token, VecPhrase,
};
pub use crate::update::update;
#[cfg(target_arch = "wasm32")]
pub use crate::wasm::*;
