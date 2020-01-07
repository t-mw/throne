use crate::rule::Rule;
use crate::state::State;
use crate::string_cache::Atom;
use crate::token::*;

pub trait OptionFilter<T> {
    fn option_filter<P: FnOnce(&T) -> bool>(self, predicate: P) -> Self;
}

// copy of nightly-only method Option::filter
impl<T> OptionFilter<T> for Option<T> {
    fn option_filter<P: FnOnce(&T) -> bool>(self, predicate: P) -> Self {
        if let Some(x) = self {
            if predicate(&x) {
                return Some(x);
            }
        }
        None
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
            .find(|m| m.atom == token.string)
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
                    if token1.string != token2.string || depth1 != depth2 {
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
    pub atom: Atom,
    pub state_i: usize,
    pub depths: (u8, u8),
    pub range: (usize, usize),
}

impl MatchLite {
    fn as_slice<'a>(&self, state: &'a State) -> &'a [Token] {
        &state[self.state_i][self.range.0..self.range.1]
    }

    pub fn to_phrase(&self, state: &State) -> Vec<Token> {
        let mut phrase = self.as_slice(state).to_vec();

        let len = phrase.len();

        if len == 1 {
            phrase[0].open_depth = 0;
            phrase[0].close_depth = 0;
        } else {
            phrase[0].open_depth -= self.depths.0;
            phrase[len - 1].close_depth -= self.depths.1;
        }

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
            atom: token.string,
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
                    .find(|m| m.atom == token.string)
            {
                if !phrase_equal(
                    &existing_match.as_slice(state),
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
                let m = MatchLite {
                    atom: token.string,
                    state_i,
                    depths: (token.open_depth, token.close_depth),
                    range: (start_i, end_i),
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
                    .find(|m| m.atom == token.string)
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

// https://stackoverflow.com/questions/44246722/is-there-any-way-to-create-an-alias-of-a-specific-fnmut
pub trait SideInput: FnMut(&Phrase) -> Option<Vec<Token>> {}
impl<F> SideInput for F where F: FnMut(&Phrase) -> Option<Vec<Token>> {}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct FirstAtoms {
    pub a0: Option<Atom>,
    pub a1: Option<Atom>,
    pub a2: Option<Atom>,
}

// Checks whether the rule's forward and backward predicates match the state.
// Returns a new rule with all variables resolved, with backwards/side
// predicates removed.
pub fn rule_matches_state<F>(r: &Rule, state: &mut State, side_input: &mut F) -> Option<Rule>
where
    F: SideInput,
{
    let inputs = &r.inputs;
    let outputs = &r.outputs;

    // per input, a list of states that could match the input
    let input_state_matches =
        if let Some(matches) = gather_potential_input_state_matches(inputs, state) {
            matches
        } else {
            return None;
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

    if permutation_count > 2000 {
        println!(
            "WARNING: rule with id {} is causing {} state permutations to be checked. Review the complexity of the rule.",
            r.id, permutation_count
        );
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

        inputs
            .iter()
            .filter(|pred| is_concrete_pred(pred) || is_var_pred(pred))
            .for_each(|v| {
                forward_concrete.push(assign_state_vars(v, state, &variables_matched));
            });

        outputs.iter().for_each(|v| {
            if is_side_pred(v) {
                let pred = assign_state_vars(v, state, &variables_matched);

                evaluate_side_pred(&pred, side_input);
            } else {
                outputs_concrete.push(assign_state_vars(v, state, &variables_matched));
            }
        });

        state.unlock_scratch();

        return Some(Rule::new(r.id, forward_concrete, outputs_concrete));
    }

    state.unlock_scratch();

    None
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
    state: &State,
) -> Option<InputStateMatches> {
    let mut potential_matches = vec![]; // inputs that could not be matched to a single state

    let mut multiple_matches = vec![]; // inputs that may yet be matched to a single state
    let mut single_matches = vec![]; // inputs that have been matched to a single state

    for (i_i, input) in inputs.iter().enumerate() {
        // TODO: exit early if we already know that side predicate or negated predicates won't match
        if is_concrete_pred(input) {
            let mut has_var = false;
            let mut states = vec![];

            if let Some(first) = extract_first_atoms_rule_input(input) {
                let start_idx = match state
                    .first_atoms
                    .binary_search_by(|probe| probe.1.cmp(&first))
                {
                    Ok(idx) => {
                        // binary search won't always find the first match,
                        // so search backwards until we find it
                        state
                            .first_atoms
                            .iter()
                            .enumerate()
                            .rev()
                            .skip(state.first_atoms.len() - 1 - idx)
                            .take_while(|(_, a)| a.1 == first)
                            .last()
                            .expect("start_idx")
                            .0
                    }
                    // error contains index that rule first atoms could be inserted at while maintaining sort order
                    Err(idx) => {
                        if idx >= state.first_atoms.len() || state.first_atoms[idx].1.a0 != first.a0
                        {
                            return None;
                        }
                        idx
                    }
                };

                state
                    .first_atoms
                    .iter()
                    .skip(start_idx)
                    .take_while(|a| {
                        first.a0 == a.1.a0
                            && (first.a1.is_none() || first.a1 == a.1.a1)
                            && (first.a2.is_none() || first.a2 == a.1.a2)
                    })
                    .for_each(|(s_i, _)| {
                        if let Some(match_has_var) =
                            test_match_without_variables(input, &state[*s_i])
                        {
                            if match_has_var {
                                has_var = true;
                            }

                            states.push(*s_i);
                        }
                    });
            } else {
                state.iter().enumerate().for_each(|(s_i, phrase_id)| {
                    if let Some(match_has_var) =
                        test_match_without_variables(input, state.get(*phrase_id))
                    {
                        if match_has_var {
                            has_var = true;
                        }

                        states.push(s_i);
                    }
                });
            };

            if states.len() == 0 {
                return None;
            }

            if states.len() == 1 {
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
        } else if input.len() == 1 && is_var_pred(&input) {
            let states = state.iter().enumerate().map(|(i, _)| i).collect::<Vec<_>>();
            potential_matches.push(InputStateMatch {
                i_i,
                has_var: true,
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

fn extract_first_atoms_rule_input(phrase: &Phrase) -> Option<FirstAtoms> {
    let to_atom = |idx| {
        phrase
            .get(idx)
            .option_filter(|t| !is_var_token(t))
            .map(|t| t.string)
    };

    let a0 = to_atom(0);
    let a1 = to_atom(1);
    let a2 = to_atom(2);

    match (a0, a1, a2) {
        (Some(_), Some(_), Some(_)) => Some(FirstAtoms { a0, a1, a2 }),
        (Some(_), Some(_), _) => Some(FirstAtoms { a0, a1, a2: None }),
        (Some(_), _, _) => Some(FirstAtoms {
            a0,
            a1: None,
            a2: None,
        }),
        _ => None,
    }
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

    for input in inputs
        .iter()
        .filter(|input| is_twoway_backwards_pred(input))
    {
        if !match_backwards_variables(input, state, variables_matched) {
            return false;
        }
    }

    for input in inputs.iter().filter(|input| is_side_pred(input)) {
        if !match_side_variables(input, state, variables_matched, side_input) {
            return false;
        }
    }

    for input in inputs
        .iter()
        .filter(|input| is_oneway_backwards_pred(input))
    {
        if !match_backwards_variables(input, state, variables_matched) {
            return false;
        }
    }

    for input in inputs.iter().filter(|input| is_negated_pred(input)) {
        // check negated predicates last, so that we know about all variables
        // from the backwards and side predicates
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
    let pred = assign_state_vars(pred, state, existing_matches_and_result);

    if let Some(eval_result) = evaluate_backwards_pred(&pred) {
        let s_i = state.len();
        state.push(eval_result);

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
    let pred = assign_state_vars(pred, state, existing_matches_and_result);

    if let Some(eval_result) = evaluate_side_pred(&pred, side_input) {
        if eval_result.len() == 0 {
            return true;
        }

        let s_i = state.len();
        state.push(eval_result);

        match_state_variables_with_existing(&pred, state, s_i, existing_matches_and_result)
    } else {
        false
    }
}

pub fn assign_state_vars(tokens: &Phrase, state: &State, matches: &[MatchLite]) -> Vec<Token> {
    let mut result: Vec<Token> = vec![];

    for token in tokens {
        if is_var_token(token) {
            if let Some(m) = matches.iter().find(|m| m.atom == token.string) {
                result.append(&mut normalize_match_phrase(token, m.to_phrase(state)));
                continue;
            }
        }

        result.push(token.clone());
    }

    result
}

pub fn assign_vars(tokens: &Phrase, matches: &[Match]) -> Vec<Token> {
    let mut result: Vec<Token> = vec![];

    for token in tokens {
        if is_var_token(token) {
            if let Some(m) = matches.iter().find(|m| m.atom == token.string) {
                result.append(&mut normalize_match_phrase(token, m.phrase.clone()));
                continue;
            }
        }

        result.push(token.clone());
    }

    result
}

pub fn evaluate_backwards_pred(tokens: &Phrase) -> Option<Vec<Token>> {
    match tokens[0].flag {
        TokenFlag::BackwardsPred(BackwardsPred::Plus) => {
            let n1 = tokens[1].as_number();
            let n2 = tokens[2].as_number();
            let n3 = tokens[3].as_number();

            match (n1, n2, n3) {
                (Some(v1), Some(v2), None) => Some(vec![
                    tokens[0].clone(),
                    tokens[1].clone(),
                    tokens[2].clone(),
                    Token::new_number(v1 + v2, 0, 1),
                ]),
                (Some(v1), None, Some(v3)) => Some(vec![
                    tokens[0].clone(),
                    tokens[1].clone(),
                    Token::new_number(v3 - v1, 0, 0),
                    tokens[3].clone(),
                ]),
                (None, Some(v2), Some(v3)) => Some(vec![
                    tokens[0].clone(),
                    Token::new_number(v3 - v2, 0, 0),
                    tokens[2].clone(),
                    tokens[3].clone(),
                ]),
                (Some(v1), Some(v2), Some(v3)) if v1 + v2 == v3 => Some(tokens.to_owned()),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::Lt) => {
            let n1 = tokens[1].as_number();
            let n2 = tokens[2].as_number();

            match (n1, n2) {
                (Some(v1), Some(v2)) if v1 < v2 => Some(tokens.to_owned()),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::Gt) => {
            let n1 = tokens[1].as_number();
            let n2 = tokens[2].as_number();

            match (n1, n2) {
                (Some(v1), Some(v2)) if v1 > v2 => Some(tokens.to_owned()),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::Lte) => {
            let n1 = tokens[1].as_number();
            let n2 = tokens[2].as_number();

            match (n1, n2) {
                (Some(v1), Some(v2)) if v1 <= v2 => Some(tokens.to_owned()),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::Gte) => {
            let n1 = tokens[1].as_number();
            let n2 = tokens[2].as_number();

            match (n1, n2) {
                (Some(v1), Some(v2)) if v1 >= v2 => Some(tokens.to_owned()),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::ModNeg) => {
            let n1 = tokens[1].as_number();
            let n2 = tokens[2].as_number();
            let n3 = tokens[3].as_number();

            let mod_neg = |x: i32, n: i32| x - n * (x / n);

            match (n1, n2, n3) {
                (Some(v1), Some(v2), None) => Some(vec![
                    tokens[0].clone(),
                    tokens[1].clone(),
                    tokens[2].clone(),
                    Token::new_number(mod_neg(v1, v2), 0, 1),
                ]),
                _ => None,
            }
        }
        TokenFlag::BackwardsPred(BackwardsPred::Equal) => {
            if phrase_equal(
                tokens.get_group(1).expect("== : first argument missing"),
                tokens.get_group(2).expect("== : second argument missing"),
                (0, 0),
                (0, 1),
            ) {
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

fn evaluate_side_pred<F>(tokens: &Phrase, side_input: &mut F) -> Option<Vec<Token>>
where
    F: SideInput,
{
    side_input(tokens)
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
