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
    let mut pred_token_iter = pred_tokens.iter();

    let mut input_depth = 0;
    let mut pred_depth = 0;

    let mut has_var = false;

    for token in input_tokens {
        let pred_token = pred_token_iter.next();

        input_depth += token.open_depth;

        if let Some(pred_token) = pred_token {
            pred_depth += pred_token.open_depth;

            let is_var = is_var_token(token);

            if !is_var {
                if token.string != pred_token.string || input_depth != pred_depth {
                    return None;
                }
            } else {
                has_var = true;

                while input_depth < pred_depth {
                    if let Some(pred_token) = pred_token_iter.next() {
                        pred_depth += pred_token.open_depth;
                        pred_depth -= pred_token.close_depth;
                    } else {
                        return None;
                    }
                }
            }

            pred_depth -= pred_token.close_depth;
        } else {
            return None;
        }

        input_depth -= token.close_depth;
    }

    if pred_token_iter.next().is_some() {
        return None;
    }

    Some(has_var)
}

#[derive(Eq, PartialEq, Debug)]
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
        }

        pred_depth -= pred_token.close_depth;
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
        }

        pred_depth -= pred_token.close_depth;
        input_depth -= token.close_depth;
    }

    true
}

// https://stackoverflow.com/questions/44246722/is-there-any-way-to-create-an-alias-of-a-specific-fnmut
pub trait SideInput: FnMut(&Phrase) -> Option<Vec<Token>> {}
impl<F> SideInput for F where F: FnMut(&Phrase) -> Option<Vec<Token>> {}

// Checks whether the rule's forward and backward predicates match the state.
// Returns a new rule with all variables resolved, with backwards/side
// predicates removed.
pub fn rule_matches_state<F>(r: &Rule, state: &mut State, side_input: &mut F) -> Option<Rule>
where
    F: SideInput,
{
    let inputs = &r.inputs;
    let outputs = &r.outputs;

    // per input, a list of states that match the input.
    let mut input_state_matches = vec![];

    for (i_i, input) in inputs.iter().enumerate() {
        // TODO: exit early if we already know that side predicate won't match
        if is_concrete_pred(input) {
            let rule_first_atoms = extract_first_atoms_rule_input(input);

            let start_idx = if let Some(first) = rule_first_atoms {
                if let Ok(idx) = state
                    .first_atoms
                    .binary_search_by(|probe| probe.1.cmp(&first))
                {
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
                } else {
                    return None;
                }
            } else {
                0
            };

            let mut matches = vec![];

            state
                .first_atoms
                .iter()
                .skip(start_idx)
                .take_while(|a| rule_first_atoms.is_none() || a.1 == rule_first_atoms.unwrap())
                .for_each(|(s_i, _)| {
                    if let Some(has_var) = test_match_without_variables(input, &state[*s_i]) {
                        matches.push((*s_i, has_var));
                    }
                });

            if matches.len() == 0 {
                return None;
            }

            input_state_matches.push((i_i, matches));
        } else if input.len() == 1 && is_var_pred(&input) {
            let matches = state
                .iter()
                .enumerate()
                .map(|(i, _)| (i, true))
                .collect::<Vec<_>>();
            input_state_matches.push((i_i, matches))
        }
    }

    // precompute values required for deriving branch indices.
    let mut input_rev_permutation_counts = vec![1; input_state_matches.len()];
    let mut permutation_count = 1;
    input_state_matches
        .iter()
        .enumerate()
        .rev()
        .for_each(|(i, (_, matches))| {
            permutation_count *= matches.len();

            if i > 0 {
                input_rev_permutation_counts[i - 1] = permutation_count;
            }
        });

    let mut variables_matched: Vec<MatchLite> = vec![];

    // we'll use state as a scratchpad for other token allocations
    let len = state.len();
    state.lock_scratch();

    'outer: for p_i in 0..permutation_count {
        variables_matched.clear();
        state.reset_scratch();

        let mut states_matched_bool = vec![false; len];

        // iterate across the graph of permutations from root to leaf, where each
        // level of the tree is an input, and each branch is a match against a state.
        // TODO: improve performance by checking inputs ordered from least to most matches
        for (concrete_input_i, (i_i, matches)) in input_state_matches.iter().enumerate() {
            let branch_idx = (p_i / input_rev_permutation_counts[concrete_input_i]) % matches.len();
            let (s_i, has_var) = matches[branch_idx];

            let input_phrase = &inputs[*i_i];

            // a previous input in this permutation has already matched the state being checked
            if input_phrase[0].is_consuming {
                if states_matched_bool[s_i] {
                    continue 'outer;
                } else {
                    states_matched_bool[s_i] = true;
                }
            }

            if has_var {
                // we know the structures are compatible from the earlier matching check
                if !match_state_variables_assuming_compatible_structure(
                    input_phrase,
                    &state,
                    s_i,
                    &mut variables_matched,
                ) {
                    continue 'outer;
                }
            }
        }

        for input in inputs.iter().filter(|input| is_backwards_pred(input)) {
            if !match_backwards_variables(input, state, &mut variables_matched) {
                continue 'outer;
            }
        }

        for input in inputs.iter().filter(|input| is_side_pred(input)) {
            if !match_side_variables(input, state, &mut variables_matched, side_input) {
                continue 'outer;
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
                    match_state_variables_with_existing(input, state, s_i, &mut variables_matched)
                })
            {
                continue 'outer;
            }
        }

        let mut forward_concrete = vec![];
        let mut outputs_concrete = vec![];

        inputs
            .iter()
            .filter(|pred| is_concrete_pred(pred))
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

fn extract_first_atoms_rule_input(phrase: &Phrase) -> Option<Atom> {
    if is_concrete_pred(phrase) {
        phrase
            .get(0)
            .option_filter(|t| !is_var_token(t))
            .map(|t| t.string)
    } else {
        None
    }
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
