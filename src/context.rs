use crate::core::Core;
use crate::matching::*;
use crate::parser;
use crate::rule::Rule;
use crate::state::{self, State};
use crate::string_cache::{Atom, StringCache};
use crate::token::*;
use crate::update::{self, update, SideInput};

use itertools::Itertools;
use rand::{self, rngs::SmallRng, SeedableRng};

use std::fmt;
use std::vec::Vec;

/// Used to build a [Context].
///
/// ```
/// # use throne::ContextBuilder;
/// let context = ContextBuilder::new()
///     .text("foo = bar")
///     .build()
///     .unwrap_or_else(|e| panic!("Failed to build throne context: {}", e));
/// ```
pub struct ContextBuilder<'a> {
    text: &'a str,
    string_cache: StringCache,
    rng: Option<&'a mut SmallRng>,
}

impl<'a> ContextBuilder<'a> {
    pub fn new() -> Self {
        ContextBuilder {
            text: "",
            string_cache: StringCache::new(),
            rng: None,
        }
    }

    /// Sets the script text used to define the initial state and rules of the [Context].
    pub fn text(mut self, text: &'a str) -> Self {
        self.text = text;
        self
    }

    /// Sets the [StringCache] used for the [Context].
    pub fn string_cache(mut self, string_cache: StringCache) -> Self {
        self.string_cache = string_cache;
        self
    }

    /// Sets the random number generator used for the [Context].
    ///
    /// Defaults to `rand::rngs::SmallRng::from_rng(&mut rand::rng())`.
    pub fn rng(mut self, rng: &'a mut SmallRng) -> Self {
        self.rng = Some(rng);
        self
    }

    /// Builds the [Context] using the provided script text to define the initial state and rules.
    /// Returns an error if the script text could not be parsed.
    pub fn build(self) -> Result<Context, parser::Error> {
        Context::new(
            self.text,
            self.string_cache,
            self.rng.unwrap_or(&mut default_rng()),
        )
    }
}

fn default_rng() -> SmallRng {
    // NB: update doc for ContextBuilder::rng if this changes
    SmallRng::from_rng(&mut rand::rng())
}

/// Stores the [State], [Rules](Rule) and [Atom] mappings for a Throne script.
///
/// Create a new `Context` using a [ContextBuilder].
#[derive(Clone)]
#[non_exhaustive]
pub struct Context {
    pub core: Core,
    pub string_cache: StringCache,
}

impl Context {
    pub(crate) fn from_text(text: &str) -> Result<Self, parser::Error> {
        ContextBuilder::new().text(text).build()
    }

    #[cfg(test)]
    pub(crate) fn from_text_rng(text: &str, rng: &mut SmallRng) -> Result<Self, parser::Error> {
        ContextBuilder::new().text(text).rng(rng).build()
    }

    fn new(
        text: &str,
        mut string_cache: StringCache,
        rng: &mut SmallRng,
    ) -> Result<Context, parser::Error> {
        let result = parser::parse(text, &mut string_cache, rng)?;

        let mut state = State::new();
        for phrase in result.state.into_iter() {
            state.push(phrase);
        }

        let qui_atom = string_cache.str_to_atom(parser::QUI);

        Ok(Context {
            core: Core {
                state,
                rules: result.rules,
                executed_rule_ids: vec![],
                rule_repeat_count: 0,
                rng: rng.clone(),
                qui_atom,
            },
            string_cache,
        })
    }

    /// Executes any [Rule] that matches the current [State] until the set of matching rules is exhausted.
    pub fn update(&mut self) -> Result<(), update::Error> {
        self.update_with_side_input(|_: &Phrase| None)
    }

    /// Equivalent to [Context::update()], but accepts a callback to respond to `^` predicates.
    pub fn update_with_side_input<F>(&mut self, side_input: F) -> Result<(), update::Error>
    where
        F: SideInput,
    {
        update(&mut self.core, side_input)
    }

    /// Executes a specific [Rule].
    ///
    /// Returns `true` if the [Rule] was successfully executed or `false` if some of its inputs could not be matched to the current [State].
    pub fn execute_rule(&mut self, rule: &Rule) -> bool {
        update::execute_rule(rule, &mut self.core.state, None)
    }

    /// Returns the set of [Rules](Rule) that may be executed in the next update.
    pub fn find_matching_rules<F>(
        &self,
        mut side_input: F,
    ) -> Result<Vec<Rule>, ExcessivePermutationError>
    where
        F: SideInput,
    {
        let state = &mut self.core.state.clone();

        let mut rules = vec![];
        for rule in &self.core.rules {
            if let Some(matching_rule) =
                rule_matches_state(&rule, state, &mut side_input)?.map(|result| result.rule)
            {
                rules.push(matching_rule);
            }
        }

        Ok(rules)
    }

    /// Alias for [StringCache::str_to_atom].
    pub fn str_to_atom(&mut self, string: &str) -> Atom {
        self.string_cache.str_to_atom(string)
    }

    /// Alias for [StringCache::str_to_existing_atom].
    pub fn str_to_existing_atom(&self, string: &str) -> Option<Atom> {
        self.string_cache.str_to_existing_atom(string)
    }

    /// Alias for [StringCache::atom_to_str].
    pub fn atom_to_str(&self, atom: Atom) -> Option<&str> {
        self.string_cache.atom_to_str(atom)
    }

    /// Alias for [StringCache::atom_to_integer].
    pub fn atom_to_integer(&self, atom: Atom) -> Option<i32> {
        StringCache::atom_to_integer(atom)
    }

    #[cfg(test)]
    pub fn with_test_rng(mut self) -> Context {
        self.core.rng = crate::tests::test_rng();
        self
    }

    /// Converts the provided text to a [Phrase] and adds it to the context's [State].
    pub fn push_state(&mut self, phrase_text: &str) {
        self.core
            .state
            .push(tokenize(phrase_text, &mut self.string_cache));
    }

    /// Copies the state from another `Context` to this one.
    pub fn extend_state_from_context(&mut self, other: &Context) {
        for phrase_id in other.core.state.iter() {
            let phrase = other.core.state.get(phrase_id);
            let new_phrase = phrase
                .iter()
                .map(|t| {
                    if StringCache::atom_to_integer(t.atom).is_some() {
                        t.clone()
                    } else {
                        let string = other
                            .string_cache
                            .atom_to_str(t.atom)
                            .expect(&format!("missing token: {:?}", t));

                        let mut new_token = t.clone();
                        new_token.atom = self.string_cache.str_to_atom(string);
                        new_token
                    }
                })
                .collect();
            self.core.state.push(new_phrase);
        }
    }

    /// Prints a representation of the `Context` to the console.
    pub fn print(&self) {
        println!("{}", self);
    }
}

impl fmt::Display for Context {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let state = state::state_to_string(&self.core.state, &self.string_cache);

        let mut rules = self
            .core
            .rules
            .iter()
            .map(|r| r.to_string(&self.string_cache))
            .collect::<Vec<_>>();
        rules.sort();

        write!(
            f,
            "state:\n{}\nrules:\n{}\n{}",
            state,
            rules.join("\n"),
            if self.core.executed_rule_ids.is_empty() {
                "no rules were executed in the previous update".to_string()
            } else {
                format!(
                    "rule ids executed in the previous update:\n{}",
                    self.core.executed_rule_ids.iter().join(", ")
                )
            }
        )
    }
}
