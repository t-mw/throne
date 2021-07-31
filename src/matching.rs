use crate::rule::{Rule, RuleBuilder};
use crate::state::State;
use crate::string_cache::Atom;
use crate::token::*;
use crate::update::SideInput;

use std::fmt;

const EXCESSIVE_PERMUTATION_LIMIT: usize = 2000;

#[derive(Debug)]
pub struct ExcessivePermutationError {
    pub rule_id: i32,
}

impl std::error::Error for ExcessivePermutationError {}

impl fmt::Display for ExcessivePermutationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,
            "Rule {} caused > {} state permutations to be checked. Review the complexity of the rule.",
            self.rule_id, EXCESSIVE_PERMUTATION_LIMIT
        )
    }
}

pub fn test_match_without_variables(input_tokens: &Phrase, pred_tokens: &Phrase) -> Option<bool> {
    let mut matcher = NoVariablesMatcher { has_var: false };

    if base_match(input_tokens, pred_tokens, &mut matcher) {
        Some(matcher.has_var)
    } else {
        None
    }
}

pub fn match_variables_twoway(
    tokens1: &Phrase,
    tokens2: &Phrase,
    existing_matches_and_result: &mut Vec<Match>,
) -> bool {
    let initial_matches_len = existing_matches_and_result.len();

    let mut matcher = TwoWayMatcher {
        existing_matches_and_result,
        initial_matches_len,
    };

    base_match(tokens1, tokens2, &mut matcher)
}

trait BaseMatcher {
    fn is_twoway_matcher(&self) -> bool;
    fn do_match(&mut self, token: &Token, phrase: &Phrase) -> bool;
}

struct NoVariablesMatcher {
    has_var: bool,
}

impl BaseMatcher for NoVariablesMatcher {
    fn is_twoway_matcher(&self) -> bool {
        false
    }

    fn do_match(&mut self, _token: &Token, _phrase: &Phrase) -> bool {
        self.has_var = true;
        true
    }
}

struct TwoWayMatcher<'a> {
    existing_matches_and_result: &'a mut Vec<Match>,
    initial_matches_len: usize,
}

impl BaseMatcher for TwoWayMatcher<'_> {
    fn is_twoway_matcher(&self) -> bool {
        true
    }

    fn do_match(&mut self, token: &Token, phrase: &Phrase) -> bool {
        let variable_already_matched = if let Some(ref existing_match) = self
            .existing_matches_and_result
            .iter()
            .find(|m| m.atom == token.atom)
        {
            if !phrase_equal(
                &existing_match.phrase[..],
                phrase,
                existing_match.depths,
                (token.open_depth, token.close_depth),
            ) {
                // this match of the variable conflicted with an existing match
                self.existing_matches_and_result
                    .drain(self.initial_matches_len..);
                return false;
            }

            true
        } else {
            false
        };

        if !variable_already_matched {
            self.existing_matches_and_result
                .push(Match::new(token, phrase.to_vec()));
        }

        true
    }
}

fn base_match(tokens1: &Phrase, tokens2: &Phrase, matcher: &mut impl BaseMatcher) -> bool {
    let mut token1_iter = tokens1.iter();
    let mut token2_iter = tokens2.iter();

    let mut depth1 = 0;
    let mut depth2 = 0;

    let mut token1_i = 0;
    let mut token2_i = 0;

    loop {
        let token1 = token1_iter.next();
        let token2 = token2_iter.next();

        match (token1, token2) {
            (None, None) => break,
            (Some(_), None) => return false,
            (None, Some(_)) => return false,
            (Some(token1), Some(token2)) => {
                depth1 += token1.open_depth;
                depth2 += token2.open_depth;

                let is_var1 = is_var_token(token1);
                let is_var2 = is_var_token(token2) && matcher.is_twoway_matcher();

                if !is_var1 && !is_var2 {
                    if token1.atom != token2.atom || depth1 != depth2 {
                        return false;
                    }

                    depth1 -= token1.close_depth;
                    depth2 -= token2.close_depth;
                } else if is_var1 && is_var2 {
                    if depth1 != depth2 {
                        return false;
                    }

                    depth1 -= token1.close_depth;
                    depth2 -= token2.close_depth;
                } else if is_var1 {
                    depth2 -= token2.close_depth;

                    // colect tokens to assign to the input variable
                    let start_i = token2_i;

                    while depth1 < depth2 {
                        if let Some(token2) = token2_iter.next() {
                            depth2 += token2.open_depth;
                            depth2 -= token2.close_depth;

                            token2_i += 1;
                        } else {
                            return false;
                        }
                    }

                    let end_i = token2_i + 1;

                    if !matcher.do_match(token1, &tokens2[start_i..end_i]) {
                        return false;
                    }

                    depth1 -= token1.close_depth;
                } else if is_var2 {
                    depth1 -= token1.close_depth;

                    // colect tokens to assign to the input variable
                    let start_i = token1_i;

                    while depth2 < depth1 {
                        if let Some(token1) = token1_iter.next() {
                            depth1 += token1.open_depth;
                            depth1 -= token1.close_depth;

                            token1_i += 1;
                        } else {
                            return false;
                        }
                    }

                    let end_i = token1_i + 1;

                    if !matcher.do_match(token2, &tokens1[start_i..end_i]) {
                        return false;
                    }

                    depth2 -= token2.close_depth;
                }

                token1_i += 1;
                token2_i += 1;
            }
        }
    }

    true
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct MatchLite {
    pub var_atom: Atom,
    pub var_open_close_depth: (u8, u8),
    pub var_open_close_depth_norm: (u8, u8),
    pub state_i: usize,
    pub state_token_range: (usize, usize),
}

impl MatchLite {
    fn as_slice<'a>(&self, state: &'a State) -> &'a [Token] {
        &state[self.state_i][self.state_token_range.0..self.state_token_range.1]
    }

    pub fn to_phrase(&self, state: &State) -> Vec<Token> {
        let mut phrase = self.as_slice(state).to_vec();

        let subset_len = phrase.len();
        let source_len = state[self.state_i].len();

        if subset_len < source_len {
            // if the phrase subset overlaps with the beginning/end of the source phrase, remove the
            // implicit open/close depth of the source phrase, since we are moving this subset into a
            // new phrase.
            if self.state_token_range.0 == 0 {
                phrase[0].open_depth -= 1;
            }
            if self.state_token_range.1 == source_len {
                phrase[subset_len - 1].close_depth -= 1;
            }
        }

        // use the variable open depth as the baseline for the new phrase subset
        phrase[0].open_depth -= self.var_open_close_depth_norm.0;

        // calculate close depth required so that sum(open_depth - close_depth) == 0
        let mut depth = 0;
        for i in 0..subset_len - 1 {
            depth += phrase[i].open_depth;
            depth -= phrase[i].close_depth;
        }
        depth += phrase[subset_len - 1].open_depth;
        phrase[subset_len - 1].close_depth = depth;

        phrase
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Match {
    pub atom: Atom,
    depths: (u8, u8),
    pub phrase: Vec<Token>,
}

impl Match {
    fn new(token: &Token, mut phrase: Vec<Token>) -> Match {
        let len = phrase.len();
        let depths = (token.open_depth, token.close_depth);

        if len == 1 {
            phrase[0].open_depth = 0;
            phrase[0].close_depth = 0;
        } else {
            phrase[0].open_depth -= depths.0;
            phrase[len - 1].close_depth -= depths.1;
        }

        Match {
            atom: token.atom,
            depths,
            phrase: phrase,
        }
    }
}

pub fn match_state_variables_with_existing(
    input_tokens: &Phrase,
    state: &State,
    s_i: usize,
    existing_matches_and_result: &mut Vec<MatchLite>,
) -> bool {
    if let Some(has_var) = test_match_without_variables(input_tokens, &state[s_i]) {
        if has_var {
            match_state_variables_assuming_compatible_structure(
                input_tokens,
                state,
                s_i,
                existing_matches_and_result,
            )
        } else {
            true
        }
    } else {
        false
    }
}

pub fn match_state_variables_assuming_compatible_structure(
    input_tokens: &Phrase,
    state: &State,
    state_i: usize,
    existing_matches_and_result: &mut Vec<MatchLite>,
) -> bool {
    let pred_tokens = &state[state_i];

    debug_assert!(test_match_without_variables(input_tokens, pred_tokens).is_some());

    let existing_matches_len = existing_matches_and_result.len();

    let mut pred_token_i = 0;

    let mut input_depth = 0;
    let mut pred_depth = 0;

    for (token_i, token) in input_tokens.iter().enumerate() {
        let pred_token = &pred_tokens[pred_token_i];
        pred_token_i += 1;

        input_depth += token.open_depth;
        pred_depth += pred_token.open_depth;

        let is_var = is_var_token(token);

        if is_var {
            pred_depth -= pred_token.close_depth;

            // colect tokens to assign to the input variable
            let start_i = pred_token_i - 1;

            while input_depth < pred_depth {
                let pred_token = &pred_tokens[pred_token_i];
                pred_token_i += 1;

                pred_depth += pred_token.open_depth;
                pred_depth -= pred_token.close_depth;
            }

            let end_i = pred_token_i;

            let variable_already_matched = if let Some(ref existing_match) =
                existing_matches_and_result
                    .iter()
                    .find(|m| m.var_atom == token.atom)
            {
                if !phrase_equal(
                    &existing_match.as_slice(state),
                    &pred_tokens[start_i..end_i],
                    existing_match.var_open_close_depth,
                    (token.open_depth, token.close_depth),
                ) {
                    // this match of the variable conflicted with an existing match
                    existing_matches_and_result.drain(existing_matches_len..);
                    return false;
                }

                true
            } else {
                false
            };

            if !variable_already_matched {
                // remove the implicit open/close depth of the source phrase
                let normalized_depths = (
                    if token_i == 0 {
                        token.open_depth - 1
                    } else {
                        token.open_depth
                    },
                    if token_i == input_tokens.len() - 1 {
                        token.close_depth - 1
                    } else {
                        token.close_depth
                    },
                );
                let m = MatchLite {
                    var_atom: token.atom,
                    var_open_close_depth: (token.open_depth, token.close_depth),
                    var_open_close_depth_norm: normalized_depths,
                    state_i,
                    state_token_range: (start_i, end_i),
                };

                existing_matches_and_result.push(m);
            }
        } else {
            pred_depth -= pred_token.close_depth;
        }

        input_depth -= token.close_depth;
    }

    true
}

pub fn match_variables_assuming_compatible_structure(
    input_tokens: &Phrase,
    pred_tokens: &Phrase,
    existing_matches_and_result: &mut Vec<Match>,
) -> bool {
    assert!(test_match_without_variables(input_tokens, pred_tokens).is_some());

    let existing_matches_len = existing_matches_and_result.len();

    let mut pred_token_i = 0;

    let mut input_depth = 0;
    let mut pred_depth = 0;

    for token in input_tokens {
        let pred_token = &pred_tokens[pred_token_i];
        pred_token_i += 1;

        input_depth += token.open_depth;
        pred_depth += pred_token.open_depth;

        let is_var = is_var_token(token);

        if is_var {
            pred_depth -= pred_token.close_depth;

            // colect tokens to assign to the input variable
            let start_i = pred_token_i - 1;

            while input_depth < pred_depth {
                let pred_token = &pred_tokens[pred_token_i];
                pred_token_i += 1;

                pred_depth += pred_token.open_depth;
                pred_depth -= pred_token.close_depth;
            }

            let end_i = pred_token_i;

            let variable_already_matched = if let Some(ref existing_match) =
                existing_matches_and_result
                    .iter()
                    .find(|m| m.atom == token.atom)
            {
                if !phrase_equal(
                    &existing_match.phrase[..],
                    &pred_tokens[start_i..end_i],
                    existing_match.depths,
                    (token.open_depth, token.close_depth),
                ) {
                    // this match of the variable conflicted with an existing match
                    existing_matches_and_result.drain(existing_matches_len..);
                    return false;
                }

                true
            } else {
                false
            };

            if !variable_already_matched {
                let phrase = pred_tokens[start_i..end_i].to_vec();
                existing_matches_and_result.push(Match::new(token, phrase));
            }
        } else {
            pred_depth -= pred_token.close_depth;
        }

        input_depth -= token.close_depth;
    }

    true
}

// Checks whether the rule's forward and backward predicates match the state.
// Returns a new rule with all variables resolved, with backwards/side
// predicates removed.
pub(crate) fn rule_matches_state<F>(
    r: &Rule,
    state: &mut State,
    side_input: &mut F,
) -> Result<Option<RuleMatchesStateResult>, ExcessivePermutationError>
where
    F: SideInput,
{
    state.update_cache();

    let inputs = &r.inputs;
    let outputs = &r.outputs;

    // per input, a list of states that could match the input
    let input_state_matches = if let Some(matches) =
        gather_potential_input_state_matches(inputs, &r.input_phrase_group_counts, state)
    {
        matches
    } else {
        return Ok(None);
    };

    // precompute values required for deriving branch indices.
    let mut input_rev_permutation_counts = vec![1; input_state_matches.potential_matches.len()];
    let mut permutation_count = 1;
    input_state_matches
        .potential_matches
        .iter()
        .enumerate()
        .rev()
        .for_each(|(i, InputStateMatch { states, .. })| {
            permutation_count *= states.len();

            if i > 0 {
                input_rev_permutation_counts[i - 1] = permutation_count;
            }
        });

    if permutation_count > EXCESSIVE_PERMUTATION_LIMIT {
        return Err(ExcessivePermutationError { rule_id: r.id });
    }

    // we'll use state as a scratchpad for other token allocations
    state.lock_scratch();

    'outer: for p_i in 0..permutation_count {
        state.reset_scratch();

        let mut variables_matched = input_state_matches.definite_matched_variables.clone();

        if !test_inputs_with_permutation(
            p_i,
            inputs,
            state,
            &input_state_matches,
            &input_rev_permutation_counts,
            &mut variables_matched,
            side_input,
        ) {
            continue 'outer;
        }

        let mut forward_concrete = vec![];
        let mut outputs_concrete = vec![];

        let mut input_phrase_group_counts = vec![];
        inputs
            .iter()
            .filter(|pred| is_concrete_pred(pred) || is_var_pred(pred))
            .for_each(|v| {
                let mut group_counter = PhraseGroupCounter::new();
                forward_concrete.push(assign_state_vars(
                    v,
                    state,
                    &variables_matched,
                    &mut group_counter,
                ));
                input_phrase_group_counts.push(group_counter.group_count);
            });

        let mut output_phrase_group_counts = vec![];
        outputs.iter().for_each(|v| {
            if is_side_pred(v) {
                let pred =
                    assign_state_vars(v, state, &variables_matched, &mut PhraseGroupCounter::new());
                side_input(&pred);
            } else {
                let mut group_counter = PhraseGroupCounter::new();
                outputs_concrete.push(assign_state_vars(
                    v,
                    state,
                    &variables_matched,
                    &mut group_counter,
                ));
                output_phrase_group_counts.push(group_counter.group_count);
            }
        });

        state.unlock_scratch();

        return Ok(Some(RuleMatchesStateResult {
            rule: RuleBuilder::new(forward_concrete, outputs_concrete, r.source_span)
                .input_phrase_group_counts(input_phrase_group_counts)
                .build(r.id),
            output_phrase_group_counts,
        }));
    }

    state.unlock_scratch();

    Ok(None)
}

#[derive(Debug)]
pub(crate) struct RuleMatchesStateResult {
    pub rule: Rule,
    pub output_phrase_group_counts: Vec<usize>,
}

#[derive(Debug)]
struct InputStateMatches {
    potential_matches: Vec<InputStateMatch>,
    definite_matched_variables: Vec<MatchLite>,
    initial_states_matched_bool: Vec<bool>,
}

#[derive(Debug)]
struct InputStateMatch {
    i_i: usize,
    has_var: bool,
    states: Vec<usize>,
}

impl InputStateMatch {
    fn test_final_match(
        &self,
        state_match_idx: usize,
        inputs: &Vec<Vec<Token>>,
        state: &State,
        states_matched_bool: &mut [bool],
        variables_matched: &mut Vec<MatchLite>,
    ) -> bool {
        let s_i = self.states[state_match_idx];
        let input_phrase = &inputs[self.i_i];

        // a previous input in the permutation has already matched the state being checked
        if input_phrase[0].is_consuming {
            if states_matched_bool[s_i] {
                return false;
            } else {
                states_matched_bool[s_i] = true;
            }
        }

        // we should know that the structures are compatible from earlier matching checks
        !self.has_var
            || match_state_variables_assuming_compatible_structure(
                input_phrase,
                state,
                s_i,
                variables_matched,
            )
    }
}

fn gather_potential_input_state_matches(
    inputs: &Vec<Vec<Token>>,
    input_phrase_group_counts: &Vec<usize>,
    state: &State,
) -> Option<InputStateMatches> {
    // only matches that have a structure that is compatible with the input should be returned from
    // this method, i.e. only variable assignments are preventing the exact matches from being known.
    let mut potential_matches = vec![]; // inputs that could not be inexpensively matched to a single state
    let mut multiple_matches = vec![]; // inputs that may yet be inexpensively matched to a single state
    let mut single_matches = vec![]; // inputs that have been matched to a single state

    for (i_i, input) in inputs.iter().enumerate() {
        if input.len() == 1 && is_var_pred(input) {
            // treat input with a single variable as a special case that can match any state
            let states = state.iter().enumerate().map(|(i, _)| i).collect::<Vec<_>>();
            potential_matches.push(InputStateMatch {
                i_i,
                has_var: true,
                states,
            });
            continue;
        }

        if !is_concrete_pred(input) && !is_var_pred(input) {
            continue;
        }

        let cached_state_matches =
            state.match_cached_state_indices_for_rule_input(input, input_phrase_group_counts[i_i]);

        let mut has_var = false;
        let mut states = vec![];
        for s_i in cached_state_matches {
            if let Some(match_has_var) = test_match_without_variables(input, &state[*s_i]) {
                if match_has_var {
                    has_var = match_has_var;
                }
                states.push(*s_i);
            }
        }

        if states.len() == 0 {
            return None;
        } else if states.len() == 1 {
            single_matches.push(InputStateMatch {
                i_i,
                has_var,
                states,
            });
        } else {
            multiple_matches.push(InputStateMatch {
                i_i,
                has_var,
                states,
            });
        }
    }

    // immediately match phrases that could only match a single state, to
    // reduce number of permutations that need to be checked later on.
    let mut definite_matched_variables = vec![];
    let mut initial_states_matched_bool = vec![false; state.len()];

    for input_state_match in &single_matches {
        if !input_state_match.test_final_match(
            0,
            inputs,
            state,
            &mut initial_states_matched_bool,
            &mut definite_matched_variables,
        ) {
            return None;
        }
    }

    // having gathered the variables for all initial single matches, eliminate
    // any other matches that have now become single matches.
    if definite_matched_variables.len() > 0 {
        for input_state_match in multiple_matches {
            let mut state_single_match_idx = None;

            if input_state_match.has_var {
                let input_phrase = &inputs[input_state_match.i_i];
                let existing_matches_len = definite_matched_variables.len();

                for (state_match_idx, s_i) in input_state_match.states.iter().enumerate() {
                    if match_state_variables_assuming_compatible_structure(
                        input_phrase,
                        state,
                        *s_i,
                        &mut definite_matched_variables,
                    ) {
                        definite_matched_variables.drain(existing_matches_len..);

                        if state_single_match_idx.is_some() {
                            state_single_match_idx = None;
                            break;
                        }
                        state_single_match_idx = Some(state_match_idx);
                    }
                }
            }

            if let Some(state_single_match_idx) = state_single_match_idx {
                if !input_state_match.test_final_match(
                    state_single_match_idx,
                    inputs,
                    state,
                    &mut initial_states_matched_bool,
                    &mut definite_matched_variables,
                ) {
                    return None;
                }
            } else {
                potential_matches.push(input_state_match);
            }
        }
    } else {
        potential_matches.append(&mut multiple_matches);
    }

    // try to improve performance later during enumeration of permutations, by
    // causing inputs to be checked from least to most matches.
    potential_matches.sort_unstable_by_key(|InputStateMatch { states, .. }| states.len());

    Some(InputStateMatches {
        potential_matches,
        definite_matched_variables,
        initial_states_matched_bool,
    })
}

fn test_inputs_with_permutation(
    p_i: usize,
    inputs: &Vec<Vec<Token>>,
    state: &mut State,
    input_state_matches: &InputStateMatches,
    input_rev_permutation_counts: &[usize],
    variables_matched: &mut Vec<MatchLite>,
    side_input: &mut impl SideInput,
) -> bool {
    let len = state.len();
    let mut states_matched_bool = input_state_matches.initial_states_matched_bool.clone();

    // iterate across the graph of permutations from root to leaf, where each
    // level of the tree is an input, and each branch is a match against a state.
    for (concrete_input_i, input_state_match) in
        input_state_matches.potential_matches.iter().enumerate()
    {
        let branch_idx =
            (p_i / input_rev_permutation_counts[concrete_input_i]) % input_state_match.states.len();

        if !input_state_match.test_final_match(
            branch_idx,
            inputs,
            state,
            &mut states_matched_bool,
            variables_matched,
        ) {
            return false;
        }
    }

    // try assigning variables from backwards predicates so that they can be used in side
    // predicates, ignoring failures because we will check again later.
    for input in inputs.iter().filter(|input| is_backwards_pred(input)) {
        match_backwards_variables(input, state, variables_matched);
    }

    for input in inputs.iter().filter(|input| is_side_pred(input)) {
        if !match_side_variables(input, state, variables_matched, side_input) {
            return false;
        }
    }

    // check all backwards predicates in order, aborting if matching fails.
    for input in inputs.iter().filter(|input| is_backwards_pred(input)) {
        if !match_backwards_variables(input, state, variables_matched) {
            return false;
        }
    }

    for input in inputs.iter().filter(|input| is_negated_pred(input)) {
        // check negated predicates last, so that we know about all variables
        // from the backwards and side predicates.
        if state
            .iter()
            .enumerate()
            .filter(|&(s_i, _)| s_i < len && !states_matched_bool[s_i])
            .any(|(s_i, _)| {
                match_state_variables_with_existing(input, state, s_i, variables_matched)
            })
        {
            return false;
        }
    }

    true
}

fn match_backwards_variables(
    pred: &Phrase,
    state: &mut State,
    existing_matches_and_result: &mut Vec<MatchLite>,
) -> bool {
    let mut group_counter = PhraseGroupCounter::new();
    let pred = assign_state_vars(pred, state, existing_matches_and_result, &mut group_counter);

    if let Some(eval_result) = evaluate_backwards_pred(&pred) {
        let s_i = state.len();
        state.push_with_metadata(eval_result, group_counter.group_count);

        match_state_variables_with_existing(&pred, state, s_i, existing_matches_and_result)
    } else {
        false
    }
}

fn match_side_variables<F>(
    pred: &Phrase,
    state: &mut State,
    existing_matches_and_result: &mut Vec<MatchLite>,
    side_input: &mut F,
) -> bool
where
    F: SideInput,
{
    let mut group_counter = PhraseGroupCounter::new();
    let pred = assign_state_vars(pred, state, existing_matches_and_result, &mut group_counter);

    if let Some(eval_result) = side_input(&pred) {
        if eval_result.len() == 0 {
            return true;
        }

        let s_i = state.len();
        state.push_with_metadata(eval_result, group_counter.group_count);

        match_state_variables_with_existing(&pred, state, s_i, existing_matches_and_result)
    } else {
        false
    }
}

pub(crate) fn assign_state_vars(
    tokens: &Phrase,
    state: &State,
    matches: &[MatchLite],
    group_counter: &mut PhraseGroupCounter,
) -> Vec<Token> {
    let mut result: Vec<Token> = vec![];

    for token in tokens {
        if is_var_token(token) {
            if let Some(m) = matches.iter().find(|m| m.var_atom == token.atom) {
                let mut append_phrase = normalize_match_phrase(token, m.to_phrase(state));
                for t in &append_phrase {
                    group_counter.count(t);
                }
                result.append(&mut append_phrase);
                continue;
            }
        }

        result.push(token.clone());
        group_counter.count(token);
    }

    // adjust depths for phrases with a single variable that matched a whole state phrase
    if result.len() == 1 && result[0].open_depth == 0 {
        result[0].open_depth = 1;
        result[0].close_depth = 1;
        group_counter.group_count += 1;
    }

    result
}

pub fn assign_vars(tokens: &Phrase, matches: &[Match]) -> Vec<Token> {
    let mut result: Vec<Token> = vec![];

    for token in tokens {
        if is_var_token(token) {
            if let Some(m) = matches.iter().find(|m| m.atom == token.atom) {
                result.append(&mut normalize_match_phrase(token, m.phrase.clone()));
                continue;
            }
        }

        result.push(token.clone());
    }

    // adjust depths for phrases with a single variable that matched a whole state phrase
    if result.len() == 1 {
        result[0].open_depth = 1;
        result[0].close_depth = 1;
    }

    result
}

pub fn evaluate_backwards_pred(tokens: &Phrase) -> Option<Vec<Token>> {
    match tokens[0].flag {
        TokenFlag::BackwardsPred(BackwardsPred::Plus) => {
            let n1 = tokens[1].as_integer();
            let n2 = tokens[2].as_integer();
            let n3 = tokens[3].as_integer();

            match (n1, n2, n3) {
                (Some(v1), Some(v2), None) => Some(vec![
                    tokens[0].clone(),
                    tokens[1].clone(),
                    tokens[2].clone(),
                    Token::new_integer(v1 + v2, 0, 1),
                ]),
                (Some(v1), None, Some(v3)) => Some(vec![
                    tokens[0].clone(),
                    tokens[1].clone(),
                    Token::new_integer(v3 - v1, 0, 0),
                    tokens[3].clone(),
                ]),
                (None, Some(v2), Some(v3)) => Some(vec![
                    tokens[0].clone(),
                    Token::new_integer(v3 - v2, 0, 0),
                    tokens[2].clone(),
                    tokens[3].clone(),
                ]),
                (Some(v1), Some(v2), Some(v3)) if v1 + v2 == v3 => Some(tokens.to_owned()),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::Minus) => {
            let n1 = tokens[1].as_integer();
            let n2 = tokens[2].as_integer();
            let n3 = tokens[3].as_integer();

            match (n1, n2, n3) {
                (Some(v1), Some(v2), None) => Some(vec![
                    tokens[0].clone(),
                    tokens[1].clone(),
                    tokens[2].clone(),
                    Token::new_integer(v1 - v2, 0, 1),
                ]),
                (Some(v1), None, Some(v3)) => Some(vec![
                    tokens[0].clone(),
                    tokens[1].clone(),
                    Token::new_integer(-v3 + v1, 0, 0),
                    tokens[3].clone(),
                ]),
                (None, Some(v2), Some(v3)) => Some(vec![
                    tokens[0].clone(),
                    Token::new_integer(v3 + v2, 0, 0),
                    tokens[2].clone(),
                    tokens[3].clone(),
                ]),
                (Some(v1), Some(v2), Some(v3)) if v1 - v2 == v3 => Some(tokens.to_owned()),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::Lt) => {
            let n1 = tokens[1].as_integer();
            let n2 = tokens[2].as_integer();

            match (n1, n2) {
                (Some(v1), Some(v2)) if v1 < v2 => Some(tokens.to_owned()),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::Gt) => {
            let n1 = tokens[1].as_integer();
            let n2 = tokens[2].as_integer();

            match (n1, n2) {
                (Some(v1), Some(v2)) if v1 > v2 => Some(tokens.to_owned()),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::Lte) => {
            let n1 = tokens[1].as_integer();
            let n2 = tokens[2].as_integer();

            match (n1, n2) {
                (Some(v1), Some(v2)) if v1 <= v2 => Some(tokens.to_owned()),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::Gte) => {
            let n1 = tokens[1].as_integer();
            let n2 = tokens[2].as_integer();

            match (n1, n2) {
                (Some(v1), Some(v2)) if v1 >= v2 => Some(tokens.to_owned()),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::ModNeg) => {
            let n1 = tokens[1].as_integer();
            let n2 = tokens[2].as_integer();
            let n3 = tokens[3].as_integer();

            match (n1, n2, n3) {
                (Some(v1), Some(v2), Some(v3)) => {
                    if v1.rem_euclid(v2) == v3 {
                        Some(tokens.to_owned())
                    } else {
                        None
                    }
                }
                (Some(v1), Some(v2), None) => Some(vec![
                    tokens[0].clone(),
                    tokens[1].clone(),
                    tokens[2].clone(),
                    Token::new_integer(v1.rem_euclid(v2), 0, 1),
                ]),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::Equal) => {
            let mut args = tokens.groups().skip(1);
            let arg1 = args.next().expect("== : first argument missing");
            let arg2 = args.next().expect("== : second argument missing");
            if phrase_equal(arg1, arg2, (0, 0), (0, 1)) {
                if tokens[0].is_negated {
                    None
                } else {
                    Some(tokens.to_owned())
                }
            } else {
                if tokens[0].is_negated {
                    Some(tokens.to_owned())
                } else {
                    None
                }
            }
        }
        _ => unreachable!("{:?}", tokens[0].flag),
    }
}

#[inline]
pub fn phrase_equal(a: &Phrase, b: &Phrase, a_depths: (u8, u8), b_depths: (u8, u8)) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let len = b.len();

    if len == 1 {
        token_equal(&a[0], &b[0], true, None, None)
    } else {
        token_equal(
            &a[0],
            &b[0],
            false,
            Some((a_depths.0, 0)),
            Some((b_depths.0, 0)),
        ) && a
            .iter()
            .skip(1)
            .take(len - 2)
            .zip(b.iter().skip(1).take(len - 2))
            .all(|(t1, t2)| token_equal(t1, t2, false, None, None))
            && token_equal(
                &a[len - 1],
                &b[len - 1],
                false,
                Some((0, a_depths.1)),
                Some((0, b_depths.1)),
            )
    }
}
