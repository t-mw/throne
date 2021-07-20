use crate::core::Core;
use crate::matching::*;
use crate::parser;
use crate::rule::Rule;
use crate::state::{self, State};
use crate::string_cache::{Atom, StringCache};
use crate::token::*;
use crate::update::{self, update};

use itertools::Itertools;
use rand::{self, rngs::SmallRng, thread_rng, SeedableRng};

use std::fmt;
use std::vec::Vec;

#[derive(Clone)]
pub struct Context {
    pub core: Core,
    pub string_cache: StringCache,
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct PhraseId {
    idx: usize,
}

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

    pub fn text(mut self, text: &'a str) -> Self {
        self.text = text;
        self
    }

    pub fn string_cache(mut self, string_cache: StringCache) -> Self {
        self.string_cache = string_cache;
        self
    }

    pub fn rng(mut self, rng: &'a mut SmallRng) -> Self {
        self.rng = Some(rng);
        self
    }

    pub fn build(self) -> Result<Context, parser::Error> {
        let mut default_rng = SmallRng::from_rng(&mut thread_rng()).unwrap();
        Context::new(
            self.text,
            self.string_cache,
            self.rng.unwrap_or(&mut default_rng),
        )
    }
}

impl Context {
    pub fn from_text(text: &str) -> Result<Self, parser::Error> {
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

        state.update_cache();
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

        self.core.state.update_cache();
    }

    pub fn str_to_atom(&mut self, text: &str) -> Atom {
        self.string_cache.str_to_atom(text)
    }

    pub fn str_to_existing_atom(&self, text: &str) -> Option<Atom> {
        self.string_cache.str_to_existing_atom(text)
    }

    pub fn atom_to_str(&self, atom: Atom) -> Option<&str> {
        self.string_cache.atom_to_str(atom)
    }

    pub fn atom_to_integer(&self, atom: Atom) -> Option<i32> {
        StringCache::atom_to_integer(atom)
    }

    #[cfg(test)]
    pub fn with_test_rng(mut self) -> Context {
        self.core.rng = crate::tests::test_rng();
        self
    }

    pub fn append_state(&mut self, text: &str) {
        self.core.state.push(tokenize(text, &mut self.string_cache));
    }

    pub fn remove_state<const N: usize>(
        &mut self,
        pattern: [Option<Atom>; N],
        match_pattern_length: bool,
    ) {
        self.core
            .state
            .remove_pattern(pattern, match_pattern_length);
    }

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

    pub fn update<F>(&mut self, side_input: F) -> Result<(), update::Error>
    where
        F: SideInput,
    {
        update(&mut self.core, side_input)
    }

    pub fn execute_rule(&mut self, rule: &Rule) {
        update::execute_rule(rule, &mut self.core.state, None);
    }

    pub fn print(&self) {
        println!("{}", self);
    }

    pub fn find_phrase<const N: usize>(
        &self,
        pattern: [Option<Atom>; N],
        match_pattern_length: bool,
    ) -> Option<&[Token]> {
        for phrase_id in self.core.state.iter() {
            let phrase = self.core.state.get(phrase_id);
            if test_phrase_pattern_match(phrase, pattern, match_pattern_length) {
                return Some(phrase);
            }
        }
        None
    }

    pub fn find_phrases<const N: usize>(
        &self,
        pattern: [Option<Atom>; N],
        match_pattern_length: bool,
    ) -> Vec<&Phrase> {
        let mut result = vec![];
        for phrase_id in self.core.state.iter() {
            let phrase = self.core.state.get(phrase_id);
            if test_phrase_pattern_match(phrase, pattern, match_pattern_length) {
                result.push(phrase);
            }
        }
        result
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
