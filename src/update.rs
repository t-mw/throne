use rand::seq::SliceRandom;

use crate::core::Core;
use crate::matching::*;
use crate::rule::Rule;
use crate::state::State;
use crate::token::*;

// https://stackoverflow.com/questions/44246722/is-there-any-way-to-create-an-alias-of-a-specific-fnmut
pub trait SideInput: FnMut(&Phrase) -> Option<Vec<Token>> {}
impl<F> SideInput for F where F: FnMut(&Phrase) -> Option<Vec<Token>> {}

pub fn update<F>(core: &mut Core, mut side_input: F)
where
    F: SideInput,
{
    let state = &mut core.state;
    let rules = &mut core.rules;
    let executed_rule_ids = &mut core.executed_rule_ids;
    let rng = &mut core.rng;

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

            if let Some(rule) = rule_matches_state(&rule, state, &mut side_input) {
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

                return;
            }
        }

        if let Some(ref matching_rule) = matching_rule {
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
