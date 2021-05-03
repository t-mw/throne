use std::fmt;

use rand::seq::SliceRandom;

use crate::core::Core;
use crate::matching;
use crate::rule::Rule;
use crate::state::State;
use crate::token::*;

const RULE_REPEAT_LIMIT: usize = 2000;

#[derive(Debug)]
pub enum Error {
    RuleRepeatError(RuleRepeatError),
    ExcessivePermutationError(matching::ExcessivePermutationError),
}

impl Error {
    pub fn rule<'a>(&self, rules: &'a [Rule]) -> Option<&'a Rule> {
        let rule_id = match self {
            Self::RuleRepeatError(e) => e.rule_id,
            Self::ExcessivePermutationError(e) => e.rule_id,
        };
        rules.iter().find(|r| r.id == rule_id)
    }
}

impl std::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::RuleRepeatError(e) => {
                write!(f, "{}", e)
            }
            Self::ExcessivePermutationError(e) => {
                write!(f, "{}", e)
            }
        }
    }
}

impl From<RuleRepeatError> for Error {
    fn from(e: RuleRepeatError) -> Self {
        Error::RuleRepeatError(e)
    }
}

impl From<matching::ExcessivePermutationError> for Error {
    fn from(e: matching::ExcessivePermutationError) -> Self {
        Error::ExcessivePermutationError(e)
    }
}

#[derive(Debug)]
pub struct RuleRepeatError {
    pub rule_id: i32,
}

impl std::error::Error for RuleRepeatError {}

impl fmt::Display for RuleRepeatError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "script execution was abandoned since rule {} appears to be repeating infinitely.",
            self.rule_id
        )
    }
}

// https://stackoverflow.com/questions/44246722/is-there-any-way-to-create-an-alias-of-a-specific-fnmut
pub trait SideInput: FnMut(&Phrase) -> Option<Vec<Token>> {}
impl<F> SideInput for F where F: FnMut(&Phrase) -> Option<Vec<Token>> {}

pub fn update<F>(core: &mut Core, mut side_input: F) -> Result<(), Error>
where
    F: SideInput,
{
    let state = &mut core.state;
    let rules = &mut core.rules;
    let executed_rule_ids = &mut core.executed_rule_ids;
    let rng = &mut core.rng;

    core.rule_repeat_count = 0;
    executed_rule_ids.clear();

    // shuffle state so that a given rule with multiple potential
    // matches does not always match the same permutation of state.
    state.shuffle(rng);

    // shuffle rules so that each has an equal chance of selection.
    rules.shuffle(rng);

    // change starting rule on each iteration to introduce randomness.
    let mut start_rule_idx = 0;

    let mut quiescence = false;

    loop {
        let mut matching_rule = None;

        if quiescence {
            state.push(vec![Token::new_atom(core.qui_atom, 0, 0)]);
        }

        state.update_first_atoms();

        for i in 0..rules.len() {
            let rule = &rules[(start_rule_idx + i) % rules.len()];

            if let Some(rule) = matching::rule_matches_state(&rule, state, &mut side_input)? {
                matching_rule = Some(rule);
                break;
            }
        }

        start_rule_idx += 1;

        if quiescence {
            quiescence = false;

            if matching_rule.is_none() {
                let qui_atom = core.qui_atom;
                assert!(
                    state
                        .iter()
                        .enumerate()
                        .filter(|&(_, p)| state.get(p)[0].atom == qui_atom)
                        .map(|(i, _)| i)
                        .collect::<Vec<_>>()
                        == vec![state.len() - 1],
                    "expected 1 * () at end of state"
                );

                let idx = state.len() - 1;
                state.remove_idx(idx);

                return Ok(());
            }
        }

        if let Some(ref matching_rule) = matching_rule {
            if let Some(previously_executed_rule_id) = executed_rule_ids.last() {
                if matching_rule.id == *previously_executed_rule_id {
                    core.rule_repeat_count += 1;
                    if core.rule_repeat_count > RULE_REPEAT_LIMIT {
                        Err(RuleRepeatError {
                            rule_id: matching_rule.id,
                        })?;
                    }
                } else {
                    core.rule_repeat_count = 0;
                }
            }

            executed_rule_ids.push(matching_rule.id);
            execute_rule(matching_rule, state);
        } else {
            quiescence = true;
        }
    }
}

pub fn execute_rule(rule: &Rule, state: &mut State) {
    let inputs = &rule.inputs;
    let outputs = &rule.outputs;

    inputs.iter().for_each(|input| {
        if input[0].is_consuming {
            state.remove_phrase(input);
        }
    });

    outputs.iter().for_each(|output| {
        state.push(output.clone());
    });
}
