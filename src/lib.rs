//! Throne is a scripting language for game prototyping and story logic.
//!
//! Documentation for learning the language itself can be found on [Github](https://github.com/t-mw/throne#reference).
//!
//! # Example
//!
//! ```
//! use throne::{ContextBuilder, tokenize};
//!
//! // Write your script text inline or in an external file included with `include_str!(..)`
//! let script = r#"
//! Mary is sister of David
//! Sarah is child of Mary
//! Tom is child of David
//!
//! CHILD is child of PARENT . AUNT is sister of PARENT .
//!     COUSIN is child of AUNT = COUSIN is cousin of CHILD
//! "#;
//!
//! // Build the Throne context using your script text to define the initial state and rules
//! let mut context = ContextBuilder::new()
//!     .text(script)
//!     .build()
//!     .unwrap_or_else(|e| panic!("Failed to build Throne context: {}", e));
//!
//! // Execute an update step
//! context.update().unwrap_or_else(|e| panic!("Throne context update failed: {}", e));
//!
//! // Fetch the updated state
//! let state = context.core.state.get_all();
//!
//! // Convert a string to a Throne phrase
//! let expected_state_phrase = tokenize("Sarah is cousin of Tom", &mut context.string_cache);
//!
//! assert_eq!(state, vec![expected_state_phrase]);
//! ```

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
pub mod token;
mod update;
#[cfg(target_arch = "wasm32")]
mod wasm;

pub use crate::context::{Context, ContextBuilder};
pub use crate::core::Core;
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub use crate::ffi::*;
pub use crate::rule::Rule;
pub use crate::state::{PhraseId, State};
pub use crate::string_cache::{Atom, StringCache};
pub use crate::token::{tokenize, Phrase, PhraseGroup, PhraseString, Token};
pub use crate::update::{update, SideInput};
#[cfg(target_arch = "wasm32")]
pub use crate::wasm::*;

pub mod errors {
    pub use crate::matching::ExcessivePermutationError;
    pub use crate::parser::Error as ParserError;
    pub use crate::update::{Error as UpdateError, RuleRepeatError};
}
