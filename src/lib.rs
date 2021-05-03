#[macro_use]
extern crate lazy_static;
extern crate pest;
#[macro_use]
extern crate pest_derive;
extern crate rand;
extern crate regex;

mod core;
#[cfg(not(target_arch = "wasm32"))]
mod ffi;
mod matching;
mod parser;
mod rule;
mod state;
mod string_cache;
mod throne;
mod token;
mod update;
#[cfg(target_arch = "wasm32")]
mod wasm;

pub use crate::core::Core;
#[cfg(not(target_arch = "wasm32"))]
pub use crate::ffi::*;
pub use crate::rule::Rule;
pub use crate::state::State;
pub use crate::string_cache::{Atom, StringCache};
pub use crate::throne::{Context, ContextBuilder, PhraseString};
pub use crate::token::{build_phrase, tokenize, Phrase, PhraseGroup, Token, VecPhrase};
pub use crate::update::update;
#[cfg(target_arch = "wasm32")]
pub use crate::wasm::*;
