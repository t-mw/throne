use rand;
use rand::{Rng, SeedableRng, StdRng};
use regex::Regex;

use std::iter::Iterator;
use std::vec::Vec;

macro_rules! dump {
    ($($a:expr),*) => ({
        let mut txt = format!("{}:{}:", file!(), line!());
        $({txt += &format!("\t{}={:?};", stringify!($a), $a)});*;
        println!("DEBUG: {}", txt);
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Token {
    pub string: String,
    is_var: bool,
    opens_group: bool,
    closes_group: bool,
}

impl Token {
    fn new(string: &str, opens_group: bool, closes_group: bool) -> Token {
        let mut chars = string.chars();
        let first_char = chars.next();
        let is_var = first_char.expect("first_char").is_ascii_uppercase()
            && chars.all(|c| c.is_numeric() || !c.is_ascii_lowercase());

        Token {
            string: string.to_string(),
            is_var,
            opens_group,
            closes_group,
        }
    }
}

pub type Phrase = Vec<Token>;
type Match = (String, Phrase);

// https://stackoverflow.com/questions/44246722/is-there-any-way-to-create-an-alias-of-a-specific-fnmut
pub trait SideInput: Fn(&Phrase) -> Option<Phrase> {}
impl<F> SideInput for F
where
    F: Fn(&Phrase) -> Option<Phrase>,
{
}

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

pub struct Context {
    rules: Vec<Rule>,
    pub state: Vec<Phrase>,
    quiescence: bool,
    rng: StdRng,
}

impl Context {
    pub fn from_text(text: &str) -> Context {
        let text = text.replace("()", "qui");
        let lines = text.split("\n");

        let parse_rule = |string: &str| {
            let mut r = string.split("=");

            let (dollars, inputs): (Vec<_>, Vec<_>) = r.next()
                .expect("r[0]")
                .split(".")
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .partition(|s| s.chars().next().expect("char") == '$');

            let outputs = r.next()
                .expect("r[1]")
                .split(".")
                .map(|s| s.trim())
                .filter(|s| !s.is_empty());

            let dollars: Vec<_> = dollars.iter().map(|s| s.split_at(1).1).collect();

            let inputs: Vec<_> = inputs
                .iter()
                .chain(dollars.iter())
                .cloned()
                .map(tokenize)
                .collect();

            let outputs = outputs.chain(dollars).map(tokenize).collect();

            return Rule::new(inputs, outputs);
        };

        let get_label = |line| {
            lazy_static! {
                static ref RE: Regex = Regex::new(r"^(#[^:]*):\s*$").unwrap();
            }

            RE.captures(line)
                .map(|caps| caps.get(1).unwrap().as_str().trim())
        };

        let get_init = |line: &String| {
            if !line.contains("=") && !line.is_empty() {
                return Some(line.split(".").map(tokenize).collect::<Vec<_>>());
            } else {
                return None;
            }
        };

        let get_rule = |line: &String| {
            if line.contains("=") && !line.is_empty() {
                return Some(parse_rule(line));
            } else {
                return None;
            }
        };

        let mut out_lines = vec![];

        let mut attach = None;

        for line in lines {
            let line = line.trim();

            if line.is_empty() {
                attach = None;
            } else {
                let label = get_label(line);

                if label.is_some() {
                    attach = label;
                } else if let Some(attach) = attach {
                    // discard the current label on quiescence
                    if line.split(|c| c == '.' || c == '=')
                        .any(|s| s.trim() == "qui")
                    {
                        out_lines.push(format!("{} . {}", attach, line));
                    } else {
                        out_lines.push(format!("{} . {} . {}", attach, line, attach));
                    }
                } else {
                    out_lines.push(String::from(line));
                }
            }
        }

        let state = out_lines
            .iter()
            .map(get_init)
            .filter(|v| v.is_some())
            .flat_map(|v| v.expect("v"))
            .collect::<Vec<_>>();

        let rules = out_lines
            .iter()
            .map(get_rule)
            .filter(|v| v.is_some())
            .map(|v| v.expect("v"))
            .collect::<Vec<_>>();

        let seed = vec![
            rand::random::<usize>(),
            rand::random::<usize>(),
            rand::random::<usize>(),
            rand::random::<usize>(),
        ];

        let rng = SeedableRng::from_seed(&seed[..]);

        Context {
            state,
            rules,
            quiescence: false,
            rng,
        }
    }

    pub fn append_state(&mut self, text: &str) {
        self.state.push(tokenize(text));
    }

    pub fn print(&self) {
        println!("state:");
        print_state(&self.state);

        println!("\nrules:");
        for r in self.rules.iter() {
            print_rule(&r);
        }
    }
}

pub fn update<F>(context: &mut Context, side_input: F)
where
    F: SideInput,
{
    let rules = &context.rules;
    let state = &mut context.state;

    loop {
        let mut matching_rule = None;

        lazy_static! {
            static ref QUI: Phrase = vec![Token::new("qui", false, false)];
        }

        if context.quiescence {
            state.push(QUI.clone());
        }

        for (_, rule) in rules.iter_rand(&mut context.rng) {
            if let Some(rule) = rule_matches_state(&rule, &state, &mut context.rng, &side_input) {
                matching_rule = Some(rule);
                break;
            }
        }

        if context.quiescence {
            context.quiescence = false;

            if matching_rule.is_none() {
                assert!(
                    state
                        .iter()
                        .enumerate()
                        .filter(|&(_, p)| **p == QUI[..])
                        .collect::<Vec<_>>()
                        == vec![(state.len() - 1, &QUI.clone())],
                    "expected 1 * () at end of state"
                );

                let idx = state.len() - 1;
                state.remove(idx);

                return;
            }
        }

        if let Some(ref matching_rule) = matching_rule {
            let inputs = &matching_rule.inputs;
            let outputs = &matching_rule.outputs;

            for input in inputs.iter() {
                let remove_idx = state.iter().position(|v| v == input);
                state.swap_remove(remove_idx.expect("remove_idx"));
            }

            for output in outputs.iter() {
                state.push(output.clone());
            }
        } else {
            context.quiescence = true;
        }
    }
}

// Checks whether the rule's forward and backward predicates match the state.
// Returns a new rule with all variables resolved, with backwards/side
// predicates removed.
fn rule_matches_state<R, F>(
    r: &Rule,
    state: &Vec<Phrase>,
    rng: &mut R,
    side_input: &F,
) -> Option<Rule>
where
    R: Rng,
    F: SideInput,
{
    let inputs = &r.inputs;
    let outputs = &r.outputs;

    let mut permutation_count = 1;

    let mut input_state_matches = Vec::with_capacity(inputs.len() * state.len());
    let mut input_state_match_start_indices = Vec::with_capacity(inputs.len());
    let mut input_state_match_counts = Vec::with_capacity(inputs.len());

    let mut input_is_backwards_pred = vec![false; inputs.len()];
    let mut input_is_side_pred = vec![false; inputs.len()];

    for (i_i, input) in inputs.iter().enumerate() {
        let mut count = 0;

        if is_backwards_pred(input) {
            input_is_backwards_pred[i_i] = true;
        } else if is_side_pred(input) {
            // TODO: exit early if we already know that side predicate won't match
            input_is_side_pred[i_i] = true;
        } else {
            for (s_i, p) in state.iter_rand(rng) {
                if match_variables_with_existing(input, p, &vec![]).is_some() {
                    input_state_matches.push(s_i);
                    count += 1;
                }
            }

            if count == 0 {
                return None;
            }

            permutation_count *= count;
        }

        input_state_match_start_indices.push(input_state_matches.len() - count);
        input_state_match_counts.push(count);
    }

    // TODO: avoid re-matching variables that are already known from a previous permutation
    let mut variables_matched = vec![];

    'outer: for p_i in 0..permutation_count {
        let mut p_i_acc = p_i;

        variables_matched.clear();

        // TODO: avoid reallocating vector
        let mut states_matched = vec![false; state.len()];

        // iterate backwards across the graph of permutations from leaf to root,
        // where each level of the tree is an input, and each branch is a match against
        // a state.
        for (i_i, input) in inputs
            .iter()
            .enumerate()
            .rev()
            .filter(|&(i_i, _)| !input_is_backwards_pred[i_i] && !input_is_side_pred[i_i])
        {
            let match_count = input_state_match_counts[i_i];

            let branch_idx = p_i_acc % match_count;
            p_i_acc = p_i_acc / match_count;

            let input_state_match_idx = input_state_match_start_indices[i_i] + branch_idx;
            let s_i = input_state_matches[input_state_match_idx];

            if states_matched[s_i] {
                continue 'outer;
            } else {
                states_matched[s_i] = true;
            }

            if let Some(ref mut result) =
                match_variables_with_existing(input, &state[s_i], &variables_matched)
            {
                variables_matched.append(result);
            } else {
                continue 'outer;
            }
        }

        for (i_i, input) in inputs.iter().enumerate() {
            let mut extra_matches = if input_is_backwards_pred[i_i] {
                match_backwards_variables(input, &variables_matched)
            } else if input_is_side_pred[i_i] {
                match_side_variables(input, &variables_matched, side_input)
            } else {
                continue;
            };

            if let Some(ref mut extra_matches) = extra_matches {
                variables_matched.append(extra_matches);
            } else {
                continue 'outer;
            }
        }

        let mut forward_concrete = vec![];
        let mut outputs_concrete = vec![];

        for v in inputs.iter() {
            if !is_backwards_pred(v) && !is_side_pred(v) {
                forward_concrete.push(assign_vars(v, &variables_matched));
            }
        }

        for v in outputs.iter() {
            outputs_concrete.push(assign_vars(v, &variables_matched));
        }

        return Some(Rule::new(forward_concrete, outputs_concrete));
    }

    return None;
}

fn match_backwards_variables(pred: &Phrase, existing_matches: &Vec<Match>) -> Option<Vec<Match>> {
    let pred = assign_vars(pred, existing_matches);

    evaluate_backwards_pred(&pred).and_then(|eval_result| {
        match_variables_with_existing(&pred, &eval_result, existing_matches)
    })
}

fn match_side_variables<F>(
    pred: &Phrase,
    existing_matches: &Vec<Match>,
    side_input: &F,
) -> Option<Vec<Match>>
where
    F: SideInput,
{
    let pred = assign_vars(pred, existing_matches);

    evaluate_side_pred(&pred, side_input).and_then(|eval_result| {
        match_variables_with_existing(&pred, &eval_result, existing_matches)
    })
}

fn assign_vars(tokens: &Phrase, matches: &Vec<Match>) -> Phrase {
    let mut result: Phrase = vec![];

    for token in tokens.iter() {
        if token.is_var {
            if let Some(&(_, ref tokens)) = matches.iter().find(|&&(ref s, _)| *s == token.string) {
                let mut tokens = tokens.clone();
                let len = tokens.len();

                if len == 1 {
                    tokens[0].opens_group = token.opens_group;
                    tokens[len - 1].closes_group = token.closes_group;
                } else {
                    tokens[0].opens_group = tokens[0].opens_group || token.opens_group;
                    tokens[len - 1].closes_group =
                        tokens[len - 1].closes_group || token.closes_group;
                }

                result.append(&mut tokens);

                continue;
            }
        }

        result.push(token.clone());
    }

    return result;
}

fn is_backwards_pred(tokens: &Phrase) -> bool {
    if tokens.len() == 0 {
        return false;
    }

    match tokens[0].string.as_str() {
        "+" => true,
        "<" => true,
        ">" => true,
        _ => false,
    }
}

fn is_side_pred(tokens: &Phrase) -> bool {
    if tokens.len() == 0 {
        return false;
    }

    match tokens[0].string.chars().next().expect("char") {
        '!' => true,
        _ => false,
    }
}

fn evaluate_backwards_pred(tokens: &Phrase) -> Option<Phrase> {
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
                    Token::new(&(v1 + v2).to_string(), false, true),
                ]),
                (Ok(v1), Err(_), Ok(v3)) => Some(vec![
                    tokens[0].clone(),
                    tokens[1].clone(),
                    Token::new(&(v3 - v1).to_string(), false, false),
                    tokens[3].clone(),
                ]),
                (Err(_), Ok(v2), Ok(v3)) => Some(vec![
                    tokens[0].clone(),
                    Token::new(&(v3 - v2).to_string(), false, false),
                    tokens[2].clone(),
                    tokens[3].clone(),
                ]),
                (Ok(v1), Ok(v2), Ok(v3)) if v1 + v2 == v3 => Some(tokens.clone()),
                _ => None,
            };
        }
        "<" => {
            use std::str::FromStr;

            let n1 = f32::from_str(&tokens[1].string);
            let n2 = f32::from_str(&tokens[2].string);

            return match (n1, n2) {
                (Ok(v1), Ok(v2)) if v1 < v2 => Some(tokens.clone()),
                _ => None,
            };
        }
        ">" => {
            use std::str::FromStr;

            let n1 = f32::from_str(&tokens[1].string);
            let n2 = f32::from_str(&tokens[2].string);

            return match (n1, n2) {
                (Ok(v1), Ok(v2)) if v1 > v2 => Some(tokens.clone()),
                _ => None,
            };
        }
        _ => None,
    }
}

fn evaluate_side_pred<F>(tokens: &Phrase, side_input: &F) -> Option<Phrase>
where
    F: SideInput,
{
    side_input(tokens)
}

fn match_variables(input_tokens: &Phrase, pred_tokens: &Phrase) -> Option<Vec<Match>> {
    match_variables_with_existing(input_tokens, pred_tokens, &vec![])
}

fn match_variables_with_existing(
    input_tokens: &Phrase,
    pred_tokens: &Phrase,
    existing_matches: &Vec<Match>,
) -> Option<Vec<Match>> {
    let mut pred_token_iter = pred_tokens.iter();
    let mut result = vec![];

    let mut input_depth = 0;
    let mut pred_depth = 0;

    for token in input_tokens.iter() {
        let pred_token = pred_token_iter.next();

        if token.opens_group {
            input_depth += 1;
        }

        if let Some(pred_token) = pred_token {
            if pred_token.opens_group {
                pred_depth += 1;
            }

            if !token.is_var {
                if token != pred_token || input_depth != pred_depth {
                    return None;
                }
            } else {
                let mut matches = vec![pred_token.clone()];

                while input_depth != pred_depth {
                    if let Some(pred_token) = pred_token_iter.next() {
                        if pred_token.opens_group {
                            pred_depth += 1;
                        }
                        if pred_token.closes_group {
                            pred_depth -= 1;
                        }

                        matches.push(pred_token.clone());
                    } else {
                        return None;
                    }
                }

                let len = matches.len();
                if len == 1 {
                    matches[0].opens_group = false;
                    matches[0].closes_group = false;
                } else {
                    matches[0].opens_group = true;
                    matches[len - 1].closes_group = true;
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

            if pred_token.closes_group {
                pred_depth -= 1;
            }
        } else {
            return None;
        }

        if token.closes_group {
            input_depth -= 1;
        }
    }

    if let Some(_) = pred_token_iter.next() {
        return None;
    }

    return Some(result);
}

fn tokenize(string: &str) -> Phrase {
    let mut string = String::from(string);

    lazy_static! {
        static ref RE1: Regex = Regex::new(r"\(\s*(\w+)\s*\)").unwrap();
    }

    loop {
        // remove instances of brackets surrounding single atoms
        let string1 = string.clone();
        let string2 = RE1.replace_all(&string1, "$1");

        if string1 == string2 {
            break;
        } else {
            string = string2.into_owned();
        }
    }

    lazy_static! {
        static ref RE2: Regex = Regex::new(r"\(|\)|\s+|[^\(\)\s]+").unwrap();
    }

    let tokens = RE2.find_iter(&string)
        .map(|m| m.as_str())
        .filter(|s| !s.trim().is_empty())
        .collect::<Vec<_>>();

    return tokenize_internal(tokens.as_slice(), 0);
}

fn tokenize_internal(tokens: &[&str], mut depth: i32) -> Phrase {
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
            let len = tokens.len();

            let opens_group = len > 1 && i == 0;
            let closes_group = len > 1 && i == tokens.len() - 1;

            result.push(Token::new(token, opens_group, closes_group));
        }
    }

    return result;
}

fn random_prime<R: Rng>(rng: &mut R) -> usize {
    #[cfg_attr(rustfmt, rustfmt_skip)]
  let primes = [
    2,    3,    5,    7,    11,   13,   17,   19,   23,   29,   31,   37,
    41,   43,   47,   53,   59,   61,   67,   71,   73,   79,   83,   89,
    97,   101,  103,  107,  109,  113,  127,  131,  137,  139,  149,  151,
    157,  163,  167,  173,  179,  181,  191,  193,  197,  199,  211,  223,
    227,  229,  233,  239,  241,  251,  257,  263,  269,  271,  277,  281,
    283,  293,  307,  311,  313,  317,  331,  337,  347,  349,  353,  359,
    367,  373,  379,  383,  389,  397,  401,  409,  419,  421,  431,  433,
    439,  443,  449,  457,  461,  463,  467,  479,  487,  491,  499,  503,
    509,  521,  523,  541,  547,  557,  563,  569,  571,  577,  587,  593,
    599,  601,  607,  613,  617,  619,  631,  641,  643,  647,  653,  659,
    661,  673,  677,  683,  691,  701,  709,  719,  727,  733,  739,  743,
    751,  757,  761,  769,  773,  787,  797,  809,  811,  821,  823,  827,
    829,  839,  853,  857,  859,  863,  877,  881,  883,  887,  907,  911,
    919,  929,  937,  941,  947,  953,  967,  971,  977,  983,  991,  997,
    1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
    1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163,
    1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249,
    1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321,
    1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439,
    1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511,
    1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601,
    1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693,
    1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783,
    1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877,
    1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987,
    1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069,
    2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143,
    2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267,
    2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347,
    2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423,
    2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543,
    2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657,
    2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713,
    2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801,
    2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903,
    2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001
  ];

    primes[rng.gen_range(0, primes.len())]
}

trait IterRand {
    type Item;

    fn iter_rand<R: Rng>(&self, rng: &mut R) -> IterRandState<Self::Item>;
}

impl<T> IterRand for [T] {
    type Item = T;

    fn iter_rand<R: Rng>(&self, rng: &mut R) -> IterRandState<T> {
        let length = self.len();
        let mut rand = 1;

        if length != 1 {
            while length % rand == 0 || rand % length == 0 {
                rand = random_prime(rng);
            }
        }

        let modulo = rand % length;

        IterRandState {
            rand,
            idx: 0,
            length,
            modulo,
            _slice: self,
        }
    }
}

impl<'a, T> Iterator for IterRandState<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<(usize, &'a T)> {
        if self.idx == self.length {
            return None;
        } else {
            let result = Some((self.modulo, &self._slice[self.modulo]));

            self.idx += 1;
            self.modulo = (self.modulo + self.rand) % self.length;

            return result;
        }
    }
}

struct IterRandState<'a, T: 'a> {
    rand: usize,
    idx: usize,
    length: usize,
    modulo: usize,
    _slice: &'a [T],
}

fn build_phrase(phrase: &Phrase) -> String {
    let mut tokens = vec![];

    for t in phrase.iter() {
        tokens.push(format!(
            "{}{}{}",
            if t.opens_group { "(" } else { "" },
            t.string,
            if t.closes_group { ")" } else { "" },
        ));
    }

    return tokens.join(" ");
}

fn print_state(state: &Vec<Phrase>) {
    for s in state.iter().map(|p| build_phrase(p)).collect::<Vec<_>>() {
        println!("{}", s);
    }
}

fn print_rule(rule: &Rule) {
    let inputs = rule.inputs
        .iter()
        .map(|p| build_phrase(p))
        .collect::<Vec<_>>()
        .join(" . ");

    let outputs = rule.outputs
        .iter()
        .map(|p| build_phrase(p))
        .collect::<Vec<_>>()
        .join(" . ");

    println!("{} = {}", inputs, outputs);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_from_text_state_test() {
        let context = Context::from_text(
            "at 0 0 wood . at 0 0 wood . #update \n\
             at 1 2 wood",
        );

        assert_eq!(
            context.state,
            [
                tokenize("at 0 0 wood"),
                tokenize("at 0 0 wood"),
                tokenize("#update"),
                tokenize("at 1 2 wood"),
            ]
        );
    }

    #[test]
    fn context_from_text_rules_test() {
        let context = Context::from_text(
            "at 0 0 wood . at 1 2 wood = at 1 0 wood \n\
             #test: \n\
             at 3 4 wood = at 5 6 wood . at 7 8 wood",
        );

        assert_eq!(
            context.rules,
            [
                Rule::new(
                    vec![tokenize("at 0 0 wood"), tokenize("at 1 2 wood")],
                    vec![tokenize("at 1 0 wood")]
                ),
                Rule::new(
                    vec![tokenize("#test"), tokenize("at 3 4 wood")],
                    vec![
                        tokenize("at 5 6 wood"),
                        tokenize("at 7 8 wood"),
                        tokenize("#test"),
                    ]
                ),
            ]
        );
    }

    #[test]
    fn context_append_state_test() {
        let mut context = Context::from_text("test 1 2");

        context.append_state("test 3 4");

        assert_eq!(
            context.state,
            vec![tokenize("test 1 2"), tokenize("test 3 4")]
        );
    }

    #[test]
    fn update_test() {
        let mut context = Context::from_text(
            "at 0 0 wood . at 0 1 wood . at 1 1 wood . at 0 1 fire . #update\n\
             #update:\n\
             at X Y wood . at X Y fire = at X Y fire\n\
             () = #spread\n\
             \n\
             #spread . $at X Y fire . + X 1 X' . + Y' 1 Y = at X' Y fire . at X Y' fire",
        );

        update(&mut context, |_: &Phrase| None);

        assert_eq!(
            context.state,
            [
                tokenize("at 0 0 wood"),
                tokenize("at 1 1 wood"),
                tokenize("at 1 1 fire"),
                tokenize("at 0 0 fire"),
                tokenize("at 0 1 fire"),
            ]
        );
    }

    #[test]
    fn token_test() {
        assert!(!Token::new("tt1", true, true).is_var);
        assert!(!Token::new("tT1", true, true).is_var);
        assert!(!Token::new("1", true, true).is_var);
        assert!(!Token::new("1Tt", true, true).is_var);
        assert!(Token::new("T", true, true).is_var);
        assert!(Token::new("TT1", true, true).is_var);
        assert!(Token::new("TT1'", true, true).is_var);
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
            (
                Rule::new(vec![tokenize("t1 T1 T2"), tokenize("t1 T1 T2")], vec![]),
                vec![tokenize("t1 0 2")],
                false,
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
            (
                Rule::new(
                    vec![
                        tokenize("+ T1 1 T2"),
                        tokenize("+ T3 1 T4"),
                        tokenize("t1 T1 T4"),
                    ],
                    vec![],
                ),
                vec![tokenize("t1 0 1")],
                true,
            ),
            (
                Rule::new(
                    vec![
                        tokenize("first"),
                        // failing backwards predicate
                        tokenize("+ 3 4 5"),
                        tokenize("at X Y fire"),
                    ],
                    vec![],
                ),
                vec![
                    tokenize("at 1 1 fire"),
                    tokenize("at 1 -1 fire"),
                    tokenize("first"),
                ],
                false,
            ),
            (
                Rule::new(vec![tokenize("at X Y fire"), tokenize("< 0 X")], vec![]),
                vec![tokenize("at 0 0 fire"), tokenize("at 2 0 fire")],
                true,
            ),
        ];

        let seed = vec![
            rand::random::<usize>(),
            rand::random::<usize>(),
            rand::random::<usize>(),
            rand::random::<usize>(),
        ];

        let mut rng: StdRng = SeedableRng::from_seed(&seed[..]);

        for (rule, state, expected) in test_cases.drain(..) {
            let result = rule_matches_state(&rule, &state, &mut rng, &|_: &Phrase| None);

            if expected {
                assert!(result.is_some());
            } else {
                assert!(result.is_none());
            }
        }
    }

    #[test]
    fn rule_matches_state_output_test() {
        let mut test_cases = vec![
            (
                Rule::new(
                    vec![
                        tokenize("t1 T1 T2"),
                        tokenize("+ T1 T2 T3'"),
                        tokenize("+ T1 T4' T2"),
                    ],
                    vec![tokenize("t3 T3'"), tokenize("t4 T4'")],
                ),
                vec![tokenize("t1 3 4")],
                Rule::new(
                    vec![tokenize("t1 3 4")],
                    vec![tokenize("t3 7"), tokenize("t4 1")],
                ),
            ),
        ];

        let seed = vec![
            rand::random::<usize>(),
            rand::random::<usize>(),
            rand::random::<usize>(),
            rand::random::<usize>(),
        ];

        let mut rng: StdRng = SeedableRng::from_seed(&seed[..]);

        for (rule, state, expected) in test_cases.drain(..) {
            let result = rule_matches_state(&rule, &state, &mut rng, &|_: &Phrase| None);

            assert!(result.is_some());
            assert_eq!(result.unwrap(), expected);
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
                    ("T3".to_string(), tokenize("t31 (t32 t33)")),
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
                Some(vec![("T2".to_string(), tokenize("t21 t22"))]),
            ),
            (
                tokenize("t1 (t21 T22 t23) T3"),
                tokenize("t1 (t21 (t221 t222 t223) t23) t3"),
                Some(vec![
                    ("T22".to_string(), tokenize("t221 t222 t223")),
                    ("T3".to_string(), tokenize("t3")),
                ]),
            ),
            (
                tokenize("t1 T2 T3"),
                tokenize("t1 t2 (t3 t2)"),
                Some(vec![
                    ("T2".to_string(), tokenize("t2")),
                    ("T3".to_string(), tokenize("t3 t2")),
                ]),
            ),
            (
                tokenize("t1 T2 T3"),
                tokenize("t1 (t2 t3) (t3 t2)"),
                Some(vec![
                    ("T2".to_string(), tokenize("t2 t3")),
                    ("T3".to_string(), tokenize("t3 t2")),
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
        assert_eq!(tokenize("t1"), [Token::new("t1", false, false)]);

        assert_eq!(
            tokenize("t1 (t21 (t221 t222 t223) t23) t3"),
            [
                Token::new("t1", true, false),
                Token::new("t21", true, false),
                Token::new("t221", true, false),
                Token::new("t222", false, false),
                Token::new("t223", false, true),
                Token::new("t23", false, true),
                Token::new("t3", false, true),
            ]
        );

        assert_eq!(
            tokenize("t1 t2 (((t3 )) t4)"),
            [
                Token::new("t1", true, false),
                Token::new("t2", false, false),
                Token::new("t3", true, false),
                Token::new("t4", false, true),
            ]
        );

        assert_eq!(
            tokenize("(t1 t2) (t3 t4)"),
            [
                Token::new("t1", true, false),
                Token::new("t2", false, true),
                Token::new("t3", true, false),
                Token::new("t4", false, true),
            ]
        );

        assert_eq!(
            tokenize("t1 t2 (t3'' t4')"),
            [
                Token::new("t1", true, false),
                Token::new("t2", false, false),
                Token::new("t3''", true, false),
                Token::new("t4'", false, true),
            ]
        );
    }
}
