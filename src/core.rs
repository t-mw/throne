use rand::{self, rngs::SmallRng};

use std::vec::Vec;

use crate::matching::*;
use crate::rule::Rule;
use crate::state::State;
use crate::string_cache::Atom;

#[derive(Clone)]
pub struct Core {
    pub state: State,
    pub rules: Vec<Rule>,
    pub executed_rule_ids: Vec<i32>,
    pub(crate) rng: SmallRng,
    pub(crate) qui_atom: Atom,
}

impl Core {
    pub fn rule_matches_state<F>(&self, rule: &Rule, mut side_input: F) -> bool
    where
        F: SideInput,
    {
        rule_matches_state(rule, &mut self.state.clone(), &mut side_input).is_some()
    }
}
