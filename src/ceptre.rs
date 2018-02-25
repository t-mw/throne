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
        let is_var = string
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit());

        Token {
            string: string.to_string(),
            depth,
            is_var,
        }
    }
}

fn match_variables(
    input_tokens: &Vec<Token>,
    pred_tokens: &Vec<Token>,
) -> Option<Vec<(String, Vec<Token>)>> {
    let mut pred_token_iter = pred_tokens.iter();
    let mut result: Vec<(String, Vec<Token>)> = vec![];

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

                let has_existing_matches = if let Some(&(_, ref existing_matches)) =
                    result.iter().find(|&&(ref t, _)| *t == token.string)
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

fn evaluate_backwards_pred(tokens: &Vec<Token>) -> Option<Vec<Token>> {
    if tokens.len() == 0 {
        return None;
    }

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
        assert!(Token::new("TT1", 0).is_var);
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

        for (input_tokens, pred_tokens, result) in test_cases.drain(..) {
            assert_eq!(match_variables(&input_tokens, &pred_tokens), result);
        }
    }

    #[test]
    fn evaluate_backwards_pred_test() {
        let mut test_cases = vec![
            (tokenize("+ A 2 3"), Some(tokenize("+ 1 2 3"))),
            (tokenize("+ 1 B 3"), Some(tokenize("+ 1 2 3"))),
            (tokenize("+ 1 2 C"), Some(tokenize("+ 1 2 3"))),
            (tokenize("+ 1 2 4"), None),
        ];

        for (input, result) in test_cases.drain(..) {
            assert_eq!(evaluate_backwards_pred(&input), result);
        }
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
