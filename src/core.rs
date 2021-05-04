use crate::matching::*;
use crate::rule::Rule;
use crate::state::State;
use crate::string_cache::Atom;

use rand::{self, rngs::SmallRng};

use std::vec::Vec;

#[derive(Clone)]
pub struct Core {
    pub state: State,
    pub rules: Vec<Rule>,
    pub executed_rule_ids: Vec<i32>,
    pub(crate) rule_repeat_count: usize,
    pub(crate) rng: SmallRng,
    pub(crate) qui_atom: Atom,
}

impl Core {
    pub fn rule_matches_state<F>(
        &self,
        rule: &Rule,
        mut side_input: F,
    ) -> Result<bool, ExcessivePermutationError>
    where
        F: SideInput,
    {
        Ok(rule_matches_state(rule, &mut self.state.clone(), &mut side_input)?.is_some())
    }
}
