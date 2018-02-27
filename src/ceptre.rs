use regex::Regex;

use std::vec::Vec;

macro_rules! dump {
    ($($a:expr),*) => ({
        let mut txt = format!("{}:{}:", file!(), line!());
        $({txt += &format!("\t{}={:?};", stringify!($a), $a)});*;
        println!("DEBUG: {}", txt);
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Token {
    string: String,
    depth: i32,
    is_var: bool,
}

impl Token {
    fn new(string: &str, depth: i32) -> Token {
        let mut chars = string.chars();
        let first_char = chars.next();
        let is_var = first_char.expect("first_char").is_ascii_uppercase()
            && chars.all(|c| c.is_numeric() || !c.is_ascii_lowercase());

        Token {
            string: string.to_string(),
            depth,
            is_var,
        }
    }

    fn set_depth(&self, depth: i32) -> Token {
        Token {
            string: self.string.clone(),
            depth,
            is_var: self.is_var,
        }
    }
}

type Phrase = Vec<Token>;
type Match = (String, Phrase);

#[derive(Debug, Eq, PartialEq)]
struct Rule {
    inputs: Vec<Phrase>,
    outputs: Vec<Phrase>,
}

impl Rule {
    fn new(inputs: Vec<Phrase>, outputs: Vec<Phrase>) -> Rule {
        Rule { inputs, outputs }
    }
}

// Checks whether the rule's forward and backward predicates match the state.
// Returns a new rule with all variables resolved, with backwards/side
// predicates removed.
fn rule_matches_state(r: &Rule, state: &Vec<Phrase>) -> Option<Rule> {
    let inputs = &r.inputs;
    let outputs = &r.outputs;

    // TODO: exit early if we already know that side predicates won't match

    let mut i_i = 0;
    let mut s_i = 0;
    let mut s_i0 = 0;
    let mut states_stack = vec![];
    let mut variables_matched = vec![];
    let mut variables_matched_length_stack = vec![];

    loop {
        let input = &inputs[i_i];
        let mut extract: Option<Vec<Match>> = None;

        let mut found = false;
        if is_backwards_pred(input) {
            // predicate does not match to specific state, so store placeholder index.
            s_i = state.len();
            found = true;
        } else {
            for (i, s) in state.iter().enumerate().skip(s_i0) {
                let result = match_variables_with_existing(input, s, &variables_matched);

                if result.is_some() && !states_stack.contains(&i) {
                    s_i = i;
                    extract = result;
                    found = true;
                    break;
                }
            }
        }

        // once variables have been matched against all forward predicates, check
        // for compatibility with backwards predicates.
        if found && i_i == inputs.len() - 1 {
            for input in inputs.iter() {
                let backwards = is_backwards_pred(input);

                if backwards {
                    if let Some(mut extra_matches) =
                        match_backwards_variables(input, &variables_matched)
                    {
                        variables_matched.append(&mut extra_matches);
                    } else {
                        found = false;
                        break;
                    }
                }
            }
        }

        if !found {
            if i_i == 0 {
                return None;
            } else {
                loop {
                    s_i0 = states_stack.pop().expect("states_stack") + 1;

                    let v_length = variables_matched_length_stack
                        .pop()
                        .expect("variables_matched_length_stack");

                    assert!(v_length <= variables_matched.len());
                    variables_matched.drain(v_length..);

                    i_i -= 1;

                    if !(s_i0 >= state.len() - 1 && states_stack.len() > 1) {
                        break;
                    }
                }

                if s_i0 >= state.len() - 1 {
                    return None;
                }
            }
        } else {
            // if predicate matches successfully, record state that we can revert to
            // if subsequent predicates match unsuccessfully.
            states_stack.push(s_i);
            s_i0 = 0;

            variables_matched_length_stack.push(variables_matched.len());
            if let Some(ref extract) = extract {
                variables_matched.append(&mut extract.clone());
            }

            i_i += 1;

            let all_matched = i_i == inputs.len();

            if all_matched {
                let mut forward_concrete = vec![];
                let mut outputs_concrete = vec![];

                for v in inputs.iter() {
                    if !is_backwards_pred(v) {
                        forward_concrete.push(assign_vars(v, &variables_matched));
                    }
                }

                for v in outputs.iter() {
                    outputs_concrete.push(assign_vars(v, &variables_matched));
                }

                return Some(Rule::new(forward_concrete, outputs_concrete));
            }
        }
    }
}

fn match_backwards_variables(
    pred: &Vec<Token>,
    existing_matches: &Vec<Match>,
) -> Option<Vec<Match>> {
    let pred = assign_vars(pred, existing_matches);

    evaluate_backwards_pred(&pred).and_then(|eval_result| {
        match_variables_with_existing(&pred, &eval_result, existing_matches)
    })
}

fn assign_vars(tokens: &Vec<Token>, matches: &Vec<Match>) -> Vec<Token> {
    let mut result: Vec<Token> = vec![];

    for token in tokens.iter() {
        if token.is_var {
            if let Some(&(_, ref tokens)) = matches.iter().find(|&&(ref s, _)| *s == token.string) {
                let var_depth = token.depth;
                let tokens = tokens.iter().map(|t| t.set_depth(t.depth + var_depth));
                result.extend(tokens);
                continue;
            }
        }

        result.push(token.clone());
    }

    return result;
}

fn is_backwards_pred(tokens: &Vec<Token>) -> bool {
    if tokens.len() == 0 {
        return false;
    }

    match tokens[0].string.as_str() {
        "+" => true,
        _ => false,
    }
}

fn evaluate_backwards_pred(tokens: &Vec<Token>) -> Option<Vec<Token>> {
    match tokens[0].string.as_str() {
        "+" => {
            use std::str::FromStr;

            let n1 = f32::from_str(&tokens[1].string);
            let n2 = f32::from_str(&tokens[2].string);
            let n3 = f32::from_str(&tokens[3].string);

            return match (n1, n2, n3) {
                (Ok(v1), Ok(v2), Err(_)) => Some(vec![
                    tokens[0].clone(),
                    tokens[1].clone(),
                    tokens[2].clone(),
                    Token::new(&(v1 + v2).to_string(), 0),
                ]),
                (Ok(v1), Err(_), Ok(v3)) => Some(vec![
                    tokens[0].clone(),
                    tokens[1].clone(),
                    Token::new(&(v3 - v1).to_string(), 0),
                    tokens[3].clone(),
                ]),
                (Err(_), Ok(v2), Ok(v3)) => Some(vec![
                    tokens[0].clone(),
                    Token::new(&(v3 - v2).to_string(), 0),
                    tokens[2].clone(),
                    tokens[3].clone(),
                ]),
                (Ok(v1), Ok(v2), Ok(v3)) if v1 + v2 == v3 => Some(tokens.clone()),
                _ => None,
            };
        }
        _ => None,
    }
}

fn match_variables(input_tokens: &Vec<Token>, pred_tokens: &Vec<Token>) -> Option<Vec<Match>> {
    match_variables_with_existing(input_tokens, pred_tokens, &vec![])
}

fn match_variables_with_existing(
    input_tokens: &Vec<Token>,
    pred_tokens: &Vec<Token>,
    existing_matches: &Vec<Match>,
) -> Option<Vec<Match>> {
    let mut pred_token_iter = pred_tokens.iter();
    let mut result = vec![];

    for token in input_tokens.iter() {
        let pred_token = pred_token_iter.next();

        if let Some(pred_token) = pred_token {
            if !token.is_var {
                if token != pred_token {
                    return None;
                }
            } else {
                let mut matches = vec![pred_token.clone()];

                // clone the iterator, because take_while consumes the final
                // non-matching item and we want to keep it. otherwise we could
                // just mutate the original iterator.
                {
                    let mut iter_clone = pred_token_iter.clone();
                    if pred_token.depth > token.depth {
                        matches.extend(iter_clone.take_while(|t| t.depth > token.depth).cloned());
                    }

                    for _ in 0..matches.len() - 1 {
                        pred_token_iter.next();
                    }
                }

                let has_existing_matches = if let Some(&(_, ref existing_matches)) = result
                    .iter()
                    .chain(existing_matches.iter())
                    .find(|&&(ref t, _)| *t == token.string)
                {
                    if *existing_matches != matches {
                        return None;
                    }

                    true
                } else {
                    false
                };

                if !has_existing_matches {
                    result.push((token.string.clone(), matches));
                }
            }
        } else {
            return None;
        }
    }

    if let Some(_) = pred_token_iter.next() {
        return None;
    }

    return Some(result);
}

fn tokenize(string: &str) -> Vec<Token> {
    tokenize_depth(string, 0)
}

fn tokenize_depth(string: &str, depth: i32) -> Vec<Token> {
    // lazy_static! {
    //     static ref RE1: Regex = Regex::new(r"^\(\s*(\w+)\s*\)$").unwrap();
    // }

    // // remove instances of brackets surrounding single terms
    // if let Some(caps) = RE1.captures(string) {
    //     string = caps.get(1).unwrap().as_str();
    // }

    lazy_static! {
        static ref RE2: Regex = Regex::new(r"\(|\)|\s+|[^\(\)\s]+").unwrap();
    }

    let tokens = RE2.find_iter(string)
        .map(|m| m.as_str())
        .filter(|s| !s.trim().is_empty())
        .collect::<Vec<_>>();

    return tokenize_internal(tokens.as_slice(), depth);
}

fn tokenize_internal(tokens: &[&str], mut depth: i32) -> Vec<Token> {
    let mut result = vec![];

    let start_depth = depth;
    let mut depth_start_i = None;

    for (i, token) in tokens.iter().enumerate() {
        if *token == "(" {
            if depth == start_depth {
                depth_start_i = Some(i + 1);
            }
            depth += 1;
            continue;
        }

        if *token == ")" {
            depth -= 1;
            if depth == start_depth {
                let range = depth_start_i.expect("depth_start_i")..i;
                result.append(&mut tokenize_internal(&tokens[range], depth + 1));
            }
            continue;
        }

        if depth == start_depth {
            result.push(Token::new(token, depth));
        }
    }

    return result;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_test() {
        assert!(!Token::new("tt1", 0).is_var);
        assert!(!Token::new("tT1", 0).is_var);
        assert!(!Token::new("1", 0).is_var);
        assert!(!Token::new("1Tt", 0).is_var);
        assert!(Token::new("T", 0).is_var);
        assert!(Token::new("TT1", 0).is_var);
        assert!(Token::new("TT1'", 0).is_var);
    }

    #[test]
    fn rule_matches_state_truthiness_test() {
        let mut test_cases = vec![
            (
                Rule::new(vec![tokenize("t1 t3 t2"), tokenize("t1 t2 t3")], vec![]),
                vec![tokenize("t1 t3 t2"), tokenize("t1 t2 t3")],
                true,
            ),
            (
                Rule::new(vec![tokenize("t1 T2 T3")], vec![]),
                vec![tokenize("t1 t2 t3")],
                true,
            ),
            (
                Rule::new(vec![tokenize("t1 T3 T2"), tokenize("t1 T2 T3")], vec![]),
                vec![tokenize("t1 t2 t3"), tokenize("t1 t2 t3")],
                false,
            ),
            (
                Rule::new(vec![tokenize("t1 T3 T2"), tokenize("t1 T2 T3")], vec![]),
                vec![tokenize("t1 t3 t2"), tokenize("t1 t2 t3")],
                true,
            ),
            (
                Rule::new(vec![tokenize("t1 T1 T2"), tokenize("+ T1 T2 T2")], vec![]),
                vec![tokenize("t1 1 2")],
                false,
            ),
            (
                Rule::new(vec![tokenize("t1 T1 T2"), tokenize("+ T1 T2 T2")], vec![]),
                vec![tokenize("t1 0 2")],
                true,
            ),
            // successful match with backwards predicates at first and last position
            (
                Rule::new(
                    vec![
                        tokenize("+ T1 T2 T2"),
                        tokenize("t1 T1 T2"),
                        tokenize("t3 T3 T4"),
                        tokenize("+ T3 T4 T2"),
                    ],
                    vec![],
                ),
                vec![tokenize("t1 0 2"), tokenize("t3 -2 4")],
                true,
            ),
            // unsuccessful match with backwards predicates at first and last position
            (
                Rule::new(
                    vec![
                        tokenize("+ T1 T2 T2"),
                        tokenize("t1 T1 T2"),
                        tokenize("t3 T3 T4"),
                        tokenize("+ T3 T4 0"),
                    ],
                    vec![],
                ),
                vec![tokenize("t1 0 2"), tokenize("t3 -2 4")],
                false,
            ),
        ];

        for (rule, state, expected) in test_cases.drain(..) {
            let result = rule_matches_state(&rule, &state);

            if expected {
                assert!(result.is_some());
            } else {
                assert!(result.is_none());
            }
        }
    }

    #[test]
    fn rule_matches_state_output_test() {
        let rule = Rule::new(
            vec![
                tokenize("t1 T1 T2"),
                tokenize("+ T1 T2 T3'"),
                tokenize("+ T1 T4' T2"),
            ],
            vec![tokenize("t3 T3'"), tokenize("t4 T4'")],
        );
        let state = vec![tokenize("t1 3 4")];
        let expected = Rule::new(
            vec![tokenize("t1 3 4")],
            vec![tokenize("t3 7"), tokenize("t4 1")],
        );

        let result = rule_matches_state(&rule, &state);

        assert!(result.is_some());
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn evaluate_backwards_pred_test() {
        let mut test_cases = vec![
            (tokenize("+ A 2 3"), Some(tokenize("+ 1 2 3"))),
            (tokenize("+ 1 B 3"), Some(tokenize("+ 1 2 3"))),
            (tokenize("+ 1 2 C"), Some(tokenize("+ 1 2 3"))),
            (tokenize("+ 1 2 4"), None),
        ];

        for (input, expected) in test_cases.drain(..) {
            assert_eq!(evaluate_backwards_pred(&input), expected);
        }
    }

    #[test]
    fn assign_vars_test() {
        let mut test_cases = vec![
            (
                tokenize("+ T1 T2 T3"),
                vec![
                    ("T1".to_string(), tokenize("1")),
                    ("T2".to_string(), tokenize("2")),
                ],
                tokenize("+ 1 2 T3"),
            ),
            (
                tokenize("T1 (T2 T3)"),
                vec![
                    ("T1".to_string(), tokenize("(t11 t12)")),
                    ("T3".to_string(), tokenize("(t31 (t32 t33))")),
                ],
                tokenize("(t11 t12) (T2 (t31 (t32 t33)))"),
            ),
        ];

        for (tokens, matches, expected) in test_cases.drain(..) {
            assert_eq!(assign_vars(&tokens, &matches), expected);
        }
    }

    #[test]
    fn match_variables_test() {
        let mut test_cases = vec![
            (
                tokenize("t1 T2 T3"),
                tokenize("t1 t2 t3"),
                Some(vec![
                    ("T2".to_string(), tokenize("t2")),
                    ("T3".to_string(), tokenize("t3")),
                ]),
            ),
            (
                tokenize("t1 T2"),
                tokenize("t1 (t21 t22)"),
                Some(vec![("T2".to_string(), tokenize_depth("t21 t22", 1))]),
            ),
            (
                tokenize("t1 (t21 T22 t23) T3"),
                tokenize("t1 (t21 (t221 t222 t223) t23) t3"),
                Some(vec![
                    ("T22".to_string(), tokenize_depth("t221 t222 t223", 2)),
                    ("T3".to_string(), tokenize("t3")),
                ]),
            ),
            (
                tokenize("t1 T2 T3"),
                tokenize("t1 t2 (t3 t2)"),
                Some(vec![
                    ("T2".to_string(), tokenize("t2")),
                    ("T3".to_string(), tokenize_depth("t3 t2", 1)),
                ]),
            ),
            (tokenize("t1 t3"), tokenize("t1 t3"), Some(vec![])),
            (tokenize("t1 t3"), tokenize("t1 (t21 t23)"), None),
            (tokenize("t1 T3"), tokenize("t1 t2 t3"), None),
            (tokenize("t1 T3 t3"), tokenize("t1 t2"), None),
            (tokenize("t1 T3 T3"), tokenize("t1 t2 t3"), None),
            (
                tokenize("t1 T3 T3"),
                tokenize("t1 t3 t3"),
                Some(vec![("T3".to_string(), tokenize("t3"))]),
            ),
        ];

        for (input_tokens, pred_tokens, expected) in test_cases.drain(..) {
            assert_eq!(match_variables(&input_tokens, &pred_tokens), expected);
        }
    }

    #[test]
    fn match_variables_with_existing_test() {
        let input_tokens = tokenize("T1 T2 T3");
        let pred_tokens = tokenize("t1 t2 t3");
        let existing_matches = vec![("T2".to_string(), tokenize("t2"))];

        let result = match_variables_with_existing(&input_tokens, &pred_tokens, &existing_matches);

        assert_eq!(
            result,
            Some(vec![
                ("T1".to_string(), tokenize("t1")),
                ("T3".to_string(), tokenize("t3")),
            ])
        )
    }

    #[test]
    fn tokenize_test() {
        assert_eq!(tokenize("t1"), [Token::new("t1", 0)]);

        assert_eq!(
            tokenize("t1 (t21 (t221 t222 t223) t23) t3"),
            [
                Token::new("t1", 0),
                Token::new("t21", 1),
                Token::new("t221", 2),
                Token::new("t222", 2),
                Token::new("t223", 2),
                Token::new("t23", 1),
                Token::new("t3", 0),
            ]
        );

        assert_eq!(
            tokenize("t1 t2 (((t3 )) t4)"),
            [
                Token::new("t1", 0),
                Token::new("t2", 0),
                Token::new("t3", 3),
                Token::new("t4", 1),
            ]
        );

        assert_eq!(
            tokenize("t1 t2 (t3'' t4')"),
            [
                Token::new("t1", 0),
                Token::new("t2", 0),
                Token::new("t3''", 1),
                Token::new("t4'", 1),
            ]
        );
    }
}
