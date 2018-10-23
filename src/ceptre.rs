use rand;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use regex::Regex;

use std::collections::HashMap;
use std::usize;
use std::vec::Vec;

macro_rules! dump {
    ($($a:expr),*) => ({
        let mut txt = format!("{}:{}:", file!(), line!());
        $({txt += &format!("\t{}={:?};", stringify!($a), $a)});*;
        println!("DEBUG: {}", txt);
    })
}

mod parser {
    #[derive(Parser)]
    #[grammar = "ceptre.pest"]
    pub struct Parser;
}

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

const MAX_NUMBER: i32 = 99999;
const MAX_STRING_IDX: usize = usize::MAX - MAX_NUMBER as usize * 2 - 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Atom {
    idx: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum BackwardsPred {
    Plus,
    Lt,
    Gt,
    Lte,
    Gte,
    ModNeg,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TokenFlag {
    None,
    Variable,
    Side,
    BackwardsPred(BackwardsPred),
}

#[derive(Clone, Debug)]
pub struct Token {
    pub string: Atom,
    is_negated: bool,
    flag: TokenFlag,
    open_depth: u8,
    close_depth: u8,
}

impl PartialEq for Token {
    fn eq(&self, other: &Token) -> bool {
        // properties not included here can be derived from string
        self.string == other.string
            && self.is_negated == other.is_negated
            && self.open_depth == other.open_depth
            && self.close_depth == other.close_depth
    }
}
impl Eq for Token {}

impl Token {
    fn new(string: &str, open_depth: u8, close_depth: u8, string_cache: &mut StringCache) -> Token {
        let mut string = string;

        let mut is_negated = false;
        let mut is_side = false;
        match string.chars().next().expect("first_char") {
            '!' => {
                is_negated = true;
                string = string.get(1..).expect("get");
            }
            '^' => {
                is_side = true;
            }
            _ => {}
        }

        let mut chars = string.chars();
        let first_char = chars.next();
        let is_var = first_char.expect("first_char").is_ascii_uppercase()
            && chars.all(|c| c.is_numeric() || !c.is_ascii_lowercase());

        let backwards_pred = match string {
            "+" => Some(BackwardsPred::Plus),
            "<" => Some(BackwardsPred::Lt),
            ">" => Some(BackwardsPred::Gt),
            "<=" => Some(BackwardsPred::Lte),
            ">=" => Some(BackwardsPred::Gte),
            "%%" => Some(BackwardsPred::ModNeg),

            _ => None,
        };

        let atom = string_cache.str_to_atom(string);

        let flag = match (is_var, is_side, backwards_pred) {
            (false, false, None) => TokenFlag::None,
            (true, false, None) => TokenFlag::Variable,
            (false, true, None) => TokenFlag::Side,
            (false, false, Some(v)) => TokenFlag::BackwardsPred(v),
            _ => unreachable!(),
        };

        Token {
            string: atom,
            flag,
            is_negated,
            open_depth,
            close_depth,
        }
    }

    pub fn as_str<'a>(&self, string_cache: &'a StringCache) -> Option<&'a str> {
        string_cache.atom_to_str(self.string)
    }

    pub fn as_number(&self) -> Option<i32> {
        StringCache::atom_to_number(self.string)
    }
}

pub type Phrase = Vec<Token>;
type Match = (Atom, Phrase);

// https://stackoverflow.com/questions/44246722/is-there-any-way-to-create-an-alias-of-a-specific-fnmut
pub trait SideInput: FnMut(&Phrase) -> Option<Phrase> {}
impl<F> SideInput for F where F: FnMut(&Phrase) -> Option<Phrase> {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rule {
    id: i32,
    inputs: Vec<Phrase>,
    outputs: Vec<Phrase>,
}

impl Rule {
    fn new(id: i32, inputs: Vec<Phrase>, outputs: Vec<Phrase>) -> Rule {
        Rule {
            id,
            inputs,
            outputs,
        }
    }
}

#[derive(Clone)]
pub struct Context {
    rules: Vec<Rule>,
    pub state: Vec<Phrase>,
    pub string_cache: StringCache,
    quiescence: bool,
    rng: SmallRng,
    first_atoms_state: Vec<FirstAtomsState>,
}

#[derive(Clone, Default)]
pub struct StringCache {
    atom_to_str: Vec<String>,
    str_to_atom: HashMap<String, Atom>,
}

impl StringCache {
    pub fn new() -> StringCache {
        Default::default()
    }

    pub fn str_to_atom(&mut self, text: &str) -> Atom {
        use std::str::FromStr;

        if let Some(n) = i32::from_str(text).ok() {
            if n.abs() > MAX_NUMBER {
                panic!("{} is large than the maximum of {}", n.abs(), MAX_NUMBER);
            }

            return Atom {
                idx: (n + MAX_NUMBER + 1) as usize + MAX_STRING_IDX,
            };
        }

        if let Some(atom) = self.str_to_existing_atom(text) {
            return atom;
        }

        let idx = self.atom_to_str.len();
        if idx > MAX_STRING_IDX {
            panic!("String cache full");
        }

        let atom = Atom { idx };

        self.atom_to_str.push(text.to_string());
        self.str_to_atom.insert(text.to_string(), atom);

        atom
    }

    pub fn str_to_existing_atom(&self, text: &str) -> Option<Atom> {
        self.str_to_atom.get(text).cloned()
    }

    fn atom_to_str(&self, atom: Atom) -> Option<&str> {
        if atom.idx <= MAX_STRING_IDX {
            Some(&self.atom_to_str[atom.idx])
        } else {
            None
        }
    }

    fn atom_to_number(atom: Atom) -> Option<i32> {
        if atom.idx <= MAX_STRING_IDX {
            None
        } else {
            Some((atom.idx - MAX_STRING_IDX) as i32 - MAX_NUMBER - 1)
        }
    }
}

impl Context {
    pub fn from_text(text: &str) -> Context {
        use pest::iterators::Pair;
        use pest::Parser;

        let text = text.replace("()", "qui");

        let file = parser::Parser::parse(parser::Rule::file, &text)
            .unwrap_or_else(|e| panic!("{}", e))
            .next()
            .unwrap();

        let mut state: Vec<Phrase> = vec![];
        let mut rules: Vec<Rule> = vec![];

        let mut string_cache = StringCache::new();

        let mut idx = -1;

        let mut pair_to_ceptre_rule = |pair: Pair<parser::Rule>, string_cache: &mut StringCache| {
            let mut pairs = pair.into_inner();
            let inputs_pair = pairs.next().unwrap();
            let outputs_pair = pairs.next().unwrap();

            idx += 1;

            let mut inputs = vec![];
            let mut outputs = vec![];
            let mut has_input_qui = false;

            for p in inputs_pair.into_inner() {
                let input_phrase = p.into_inner().next().unwrap();

                match input_phrase.as_rule() {
                    parser::Rule::copy_phrase => {
                        let copy_phrase = tokenize(
                            input_phrase.into_inner().next().unwrap().as_str(),
                            string_cache,
                        );

                        inputs.push(copy_phrase.clone());
                        outputs.push(copy_phrase);
                    }
                    rule_type => {
                        has_input_qui = has_input_qui || rule_type == parser::Rule::qui;
                        inputs.push(tokenize(input_phrase.as_str(), string_cache));
                    }
                }
            }

            for p in outputs_pair.into_inner() {
                let output_phrase = p.into_inner().next().unwrap();

                match output_phrase.as_rule() {
                    parser::Rule::qui => (),
                    _ => outputs.push(tokenize(output_phrase.as_str(), string_cache)),
                }
            }

            (Rule::new(idx, inputs, outputs), has_input_qui)
        };

        for line in file.into_inner() {
            match line.as_rule() {
                parser::Rule::stage => {
                    let mut stage = line.into_inner();
                    let phrase_pair = stage.next().unwrap();
                    let stage_phrase = tokenize(phrase_pair.as_str(), &mut string_cache);

                    for pair in stage {
                        let (mut r, has_input_qui) = pair_to_ceptre_rule(pair, &mut string_cache);

                        // insert at beginning, so that 'first atoms' optimization is effective
                        r.inputs.insert(0, stage_phrase.clone());

                        if !has_input_qui {
                            r.outputs.push(stage_phrase.clone());
                        }

                        rules.push(r);
                    }
                }
                parser::Rule::rule => {
                    rules.push(pair_to_ceptre_rule(line, &mut string_cache).0);
                }
                parser::Rule::state => for phrase in line.into_inner() {
                    state.push(tokenize(phrase.as_str(), &mut string_cache));
                },
                parser::Rule::EOI => (),
                _ => unreachable!("{}", line),
            }
        }

        let seed = [
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
            rand::random::<u8>(),
        ];

        let rng = SmallRng::from_seed(seed);
        let first_atoms_state = extract_first_atoms_state(&state);

        Context {
            state,
            rules,
            string_cache,
            quiescence: false,
            rng,
            first_atoms_state,
        }
    }

    pub fn str_to_atom(&mut self, text: &str) -> Atom {
        self.string_cache.str_to_atom(text)
    }

    pub fn str_to_existing_atom(&self, text: &str) -> Option<Atom> {
        self.string_cache.str_to_existing_atom(text)
    }

    pub fn with_test_rng(mut self) -> Context {
        self.rng = test_rng();

        self
    }

    pub fn append_state(&mut self, text: &str) {
        self.state.push(tokenize(text, &mut self.string_cache));
    }

    pub fn find_matching_rules(&mut self) -> Vec<Rule> {
        let state = &self.state;
        let string_cache = &mut self.string_cache;
        let first_atoms_state = &self.first_atoms_state;

        self.rules
            .iter()
            .filter_map(|rule| {
                rule_matches_state(
                    &rule,
                    state,
                    &mut |_: &Phrase| None,
                    string_cache,
                    first_atoms_state,
                )
            }).collect()
    }

    pub fn execute_rule(&mut self, rule: &Rule) {
        let inputs = &rule.inputs;
        let outputs = &rule.outputs;

        inputs.iter().for_each(|input| {
            let remove_idx = self.state.iter().position(|v| v == input);
            self.state.swap_remove(remove_idx.expect("remove_idx"));
        });

        outputs.iter().for_each(|output| {
            self.state.push(output.clone());
        });
    }

    pub fn print(&self) {
        println!("state:");
        print_state(&self.state, &self.string_cache);

        println!("\nrules:");
        let mut rules = self
            .rules
            .iter()
            .map(|r| rule_to_string(r, &self.string_cache))
            .collect::<Vec<_>>();
        rules.sort();

        rules.iter().for_each(|r| {
            println!("{}", r);
        });
    }

    pub fn find_phrase<'a>(&'a self, s1: Option<&str>) -> Option<&'a Phrase> {
        self.find_phrase2(s1, None)
    }

    pub fn find_phrase2<'a>(&'a self, s1: Option<&str>, s2: Option<&str>) -> Option<&'a Phrase> {
        self.find_phrase3(s1, s2, None)
    }

    pub fn find_phrase3<'a>(
        &'a self,
        s1: Option<&str>,
        s2: Option<&str>,
        s3: Option<&str>,
    ) -> Option<&'a Phrase> {
        self.find_phrase4(s1, s2, s3, None)
    }

    pub fn find_phrase4<'a>(
        &'a self,
        s1: Option<&str>,
        s2: Option<&str>,
        s3: Option<&str>,
        s4: Option<&str>,
    ) -> Option<&'a Phrase> {
        self.find_phrase5(s1, s2, s3, s4, None)
    }

    pub fn find_phrase5<'a>(
        &'a self,
        s1: Option<&str>,
        s2: Option<&str>,
        s3: Option<&str>,
        s4: Option<&str>,
        s5: Option<&str>,
    ) -> Option<&'a Phrase> {
        let mut atom1 = None;
        let mut atom2 = None;
        let mut atom3 = None;
        let mut atom4 = None;
        let mut atom5 = None;

        if let Some(s) = s1 {
            if let Some(atom) = self.str_to_existing_atom(s) {
                atom1 = Some(atom);
            } else {
                return None;
            }
        };

        if let Some(s) = s2 {
            if let Some(atom) = self.str_to_existing_atom(s) {
                atom2 = Some(atom);
            } else {
                return None;
            }
        };

        if let Some(s) = s3 {
            if let Some(atom) = self.str_to_existing_atom(s) {
                atom3 = Some(atom);
            } else {
                return None;
            }
        };

        if let Some(s) = s4 {
            if let Some(atom) = self.str_to_existing_atom(s) {
                atom4 = Some(atom);
            } else {
                return None;
            }
        };

        if let Some(s) = s5 {
            if let Some(atom) = self.str_to_existing_atom(s) {
                atom5 = Some(atom);
            } else {
                return None;
            }
        };

        for p in &self.state {
            match (
                p.get(0).map(|t| &t.string),
                p.get(1).map(|t| &t.string),
                p.get(2).map(|t| &t.string),
                p.get(3).map(|t| &t.string),
                p.get(4).map(|t| &t.string),
            ) {
                (s1, s2, s3, s4, s5)
                    if (atom1.is_none() || s1 == atom1.as_ref())
                        && (atom2.is_none() || s2 == atom2.as_ref())
                        && (atom3.is_none() || s3 == atom3.as_ref())
                        && (atom4.is_none() || s4 == atom4.as_ref())
                        && (atom5.is_none() || s5 == atom5.as_ref()) =>
                {
                    return Some(p);
                }
                _ => (),
            };
        }

        None
    }

    pub fn find_phrases<'a>(&'a self, s1: Option<&str>) -> Vec<&'a Phrase> {
        self.find_phrases2(s1, None)
    }

    pub fn find_phrases2<'a>(&'a self, s1: Option<&str>, s2: Option<&str>) -> Vec<&'a Phrase> {
        self.find_phrases3(s1, s2, None)
    }

    pub fn find_phrases3<'a>(
        &'a self,
        s1: Option<&str>,
        s2: Option<&str>,
        s3: Option<&str>,
    ) -> Vec<&'a Phrase> {
        self.find_phrases4(s1, s2, s3, None)
    }

    pub fn find_phrases4<'a>(
        &'a self,
        s1: Option<&str>,
        s2: Option<&str>,
        s3: Option<&str>,
        s4: Option<&str>,
    ) -> Vec<&'a Phrase> {
        self.find_phrases5(s1, s2, s3, s4, None)
    }

    pub fn find_phrases5<'a>(
        &'a self,
        s1: Option<&str>,
        s2: Option<&str>,
        s3: Option<&str>,
        s4: Option<&str>,
        s5: Option<&str>,
    ) -> Vec<&'a Phrase> {
        let mut atom1 = None;
        let mut atom2 = None;
        let mut atom3 = None;
        let mut atom4 = None;
        let mut atom5 = None;

        if let Some(s) = s1 {
            if let Some(atom) = self.str_to_existing_atom(s) {
                atom1 = Some(atom);
            } else {
                return vec![];
            }
        };

        if let Some(s) = s2 {
            if let Some(atom) = self.str_to_existing_atom(s) {
                atom2 = Some(atom);
            } else {
                return vec![];
            }
        };

        if let Some(s) = s3 {
            if let Some(atom) = self.str_to_existing_atom(s) {
                atom3 = Some(atom);
            } else {
                return vec![];
            }
        };

        if let Some(s) = s4 {
            if let Some(atom) = self.str_to_existing_atom(s) {
                atom4 = Some(atom);
            } else {
                return vec![];
            }
        };

        if let Some(s) = s5 {
            if let Some(atom) = self.str_to_existing_atom(s) {
                atom5 = Some(atom);
            } else {
                return vec![];
            }
        };

        self.state
            .iter()
            .filter(|p| {
                match (
                    p.get(0).map(|t| &t.string),
                    p.get(1).map(|t| &t.string),
                    p.get(2).map(|t| &t.string),
                    p.get(3).map(|t| &t.string),
                    p.get(4).map(|t| &t.string),
                ) {
                    (s1, s2, s3, s4, s5) => {
                        (atom1.is_none() || s1 == atom1.as_ref())
                            && (atom2.is_none() || s2 == atom2.as_ref())
                            && (atom3.is_none() || s3 == atom3.as_ref())
                            && (atom4.is_none() || s4 == atom4.as_ref())
                            && (atom5.is_none() || s5 == atom5.as_ref())
                    }
                }
            }).collect()
    }
}

pub fn update<F>(context: &mut Context, mut side_input: F)
where
    F: SideInput,
{
    let qui: Phrase = vec![Token::new("qui", 0, 0, &mut context.string_cache)];

    // shuffle state so that a given rule with multiple potential
    // matches does not always match the same permutation of state.
    context.rng.shuffle(&mut context.state);

    // shuffle rules so that each has an equal chance of selection.
    context.rng.shuffle(&mut context.rules);

    // change starting rule on each iteration to introduce randomness.
    let mut start_rule_idx = 0;

    loop {
        let mut matching_rule = None;

        if context.quiescence {
            context.state.push(qui.clone());
        }

        context.first_atoms_state = extract_first_atoms_state(&context.state);
        {
            let rules = &context.rules;
            let state = &context.state;

            for i in 0..rules.len() {
                let rule = &rules[(start_rule_idx + i) % rules.len()];

                if let Some(rule) = rule_matches_state(
                    &rule,
                    &state,
                    &mut side_input,
                    &mut context.string_cache,
                    &context.first_atoms_state,
                ) {
                    matching_rule = Some(rule);
                    break;
                }
            }

            start_rule_idx += 1;
        }

        if context.quiescence {
            context.quiescence = false;

            if matching_rule.is_none() {
                let state = &mut context.state;

                assert!(
                    state
                        .iter()
                        .enumerate()
                        .filter(|&(_, p)| **p == qui[..])
                        .collect::<Vec<_>>()
                        == vec![(state.len() - 1, &qui.clone())],
                    "expected 1 * () at end of state"
                );

                let idx = state.len() - 1;
                state.remove(idx);

                return;
            }
        }

        if let Some(ref matching_rule) = matching_rule {
            context.execute_rule(matching_rule);
        } else {
            context.quiescence = true;
        }
    }
}

// Checks whether the rule's forward and backward predicates match the state.
// Returns a new rule with all variables resolved, with backwards/side
// predicates removed.
fn rule_matches_state<F>(
    r: &Rule,
    state: &[Phrase],
    side_input: &mut F,
    string_cache: &mut StringCache,
    state_first_atoms: &[FirstAtomsState],
) -> Option<Rule>
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
                if let Ok(idx) = state_first_atoms.binary_search_by(|probe| probe.1.cmp(&first)) {
                    // binary search won't always find the first match,
                    // so search backwards until we find it
                    state_first_atoms
                        .iter()
                        .enumerate()
                        .rev()
                        .skip(state_first_atoms.len() - 1 - idx)
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

            state_first_atoms
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

    'outer: for p_i in 0..permutation_count {
        let mut variables_matched = vec![];
        let mut states_matched_bool = vec![false; state.len()];

        // iterate across the graph of permutations from root to leaf, where each
        // level of the tree is an input, and each branch is a match against a state.
        for (c_i, (i_i, matches)) in input_state_matches.iter().enumerate() {
            let branch_idx = (p_i / input_rev_permutation_counts[c_i]) % matches.len();
            let (s_i, has_var) = matches[branch_idx];

            // a previous input in this permutation has already matched the state being checked
            if states_matched_bool[s_i] {
                continue 'outer;
            } else {
                states_matched_bool[s_i] = true;
            }

            if has_var {
                // we know the structures are compatible from the earlier matching check
                if let Some(ref mut result) = match_variables_assuming_compatible_structure(
                    &inputs[*i_i],
                    &state[s_i],
                    &variables_matched,
                ) {
                    variables_matched.append(result);
                } else {
                    continue 'outer;
                }
            }
        }

        for input in inputs.iter().filter(|input| is_backwards_pred(input)) {
            let mut extra_matches =
                match_backwards_variables(input, &variables_matched, string_cache);

            if let Some(ref mut extra_matches) = extra_matches {
                variables_matched.append(extra_matches);
            } else {
                continue 'outer;
            }
        }

        for input in inputs.iter().filter(|input| is_side_pred(input)) {
            let mut extra_matches = match_side_variables(input, &variables_matched, side_input);

            if let Some(ref mut extra_matches) = extra_matches {
                variables_matched.append(extra_matches);
            } else {
                continue 'outer;
            }
        }

        for input in inputs.iter().filter(|input| is_negated_pred(input)) {
            // check negated predicates last, so that we know about all variables
            // from the backwards and side predicates
            if state
                .iter()
                .enumerate()
                .filter(|&(s_i, _)| !states_matched_bool[s_i])
                .any(|(_, s)| match_variables_with_existing(input, s, &variables_matched).is_some())
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
                forward_concrete.push(assign_vars(v, &variables_matched));
            });

        outputs.iter().for_each(|v| {
            if is_side_pred(v) {
                let pred = assign_vars(v, &variables_matched);

                evaluate_side_pred(&pred, side_input);
            } else {
                outputs_concrete.push(assign_vars(v, &variables_matched));
            }
        });

        return Some(Rule::new(r.id, forward_concrete, outputs_concrete));
    }

    None
}

fn match_backwards_variables(
    pred: &Phrase,
    existing_matches: &[Match],
    string_cache: &mut StringCache,
) -> Option<Vec<Match>> {
    let pred = assign_vars(pred, existing_matches);

    evaluate_backwards_pred(&pred, string_cache).and_then(|eval_result| {
        match_variables_with_existing(&pred, &eval_result, existing_matches)
    })
}

fn match_side_variables<F>(
    pred: &Phrase,
    existing_matches: &[Match],
    side_input: &mut F,
) -> Option<Vec<Match>>
where
    F: SideInput,
{
    let pred = assign_vars(pred, existing_matches);

    evaluate_side_pred(&pred, side_input).and_then(|eval_result| {
        match_variables_with_existing(&pred, &eval_result, existing_matches)
    })
}

fn assign_vars(tokens: &Phrase, matches: &[Match]) -> Phrase {
    let mut result: Phrase = vec![];

    for token in tokens {
        if is_var_token(token) {
            if let Some(&(_, ref tokens)) = matches.iter().find(|&&(ref s, _)| *s == token.string) {
                let mut tokens = tokens.to_owned();
                let len = tokens.len();

                if len == 1 {
                    tokens[0].is_negated = token.is_negated;
                    tokens[0].open_depth = token.open_depth;
                    tokens[len - 1].close_depth = token.close_depth;
                } else {
                    tokens[0].is_negated = token.is_negated;
                    if token.open_depth > 0 {
                        tokens[0].open_depth += token.open_depth
                    }
                    if token.close_depth > 0 {
                        tokens[len - 1].close_depth += token.close_depth
                    }
                }

                result.append(&mut tokens);

                continue;
            }
        }

        result.push(token.clone());
    }

    result
}

fn is_var_token(token: &Token) -> bool {
    token.flag == TokenFlag::Variable
}

fn is_backwards_pred(tokens: &Phrase) -> bool {
    if let TokenFlag::BackwardsPred(_) = tokens[0].flag {
        true
    } else {
        false
    }
}

fn is_side_pred(tokens: &Phrase) -> bool {
    tokens[0].flag == TokenFlag::Side
}

fn is_negated_pred(tokens: &Phrase) -> bool {
    tokens[0].is_negated
}

fn is_concrete_pred(tokens: &Phrase) -> bool {
    !is_negated_pred(tokens) && tokens[0].flag == TokenFlag::None
}

fn evaluate_backwards_pred(tokens: &Phrase, string_cache: &mut StringCache) -> Option<Phrase> {
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
                    Token::new(&(v1 + v2).to_string(), 0, 1, string_cache),
                ]),
                (Some(v1), None, Some(v3)) => Some(vec![
                    tokens[0].clone(),
                    tokens[1].clone(),
                    Token::new(&(v3 - v1).to_string(), 0, 0, string_cache),
                    tokens[3].clone(),
                ]),
                (None, Some(v2), Some(v3)) => Some(vec![
                    tokens[0].clone(),
                    Token::new(&(v3 - v2).to_string(), 0, 0, string_cache),
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
                    Token::new(&mod_neg(v1, v2).to_string(), 0, 1, string_cache),
                ]),
                _ => None,
            }
        }
        _ => unreachable!(),
    }
}

fn evaluate_side_pred<F>(tokens: &Phrase, side_input: &mut F) -> Option<Phrase>
where
    F: SideInput,
{
    side_input(tokens)
}

fn test_match_without_variables(input_tokens: &Phrase, pred_tokens: &Phrase) -> Option<bool> {
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

fn match_variables_with_existing(
    input_tokens: &Phrase,
    pred_tokens: &Phrase,
    existing_matches: &[Match],
) -> Option<Vec<Match>> {
    if let Some(has_var) = test_match_without_variables(input_tokens, pred_tokens) {
        if has_var {
            match_variables_assuming_compatible_structure(
                input_tokens,
                pred_tokens,
                existing_matches,
            )
        } else {
            Some(vec![])
        }
    } else {
        None
    }
}

fn match_variables_assuming_compatible_structure(
    input_tokens: &Phrase,
    pred_tokens: &Phrase,
    existing_matches: &[Match],
) -> Option<Vec<Match>> {
    assert!(test_match_without_variables(input_tokens, pred_tokens).is_some());

    let mut result = vec![];
    let mut pred_token_iter = pred_tokens.iter();

    let mut input_depth = 0;
    let mut pred_depth = 0;

    for token in input_tokens {
        let pred_token = pred_token_iter.next().expect("pred_token");

        input_depth += token.open_depth;
        pred_depth += pred_token.open_depth;

        let is_var = is_var_token(token);

        if is_var {
            // colect tokens to assign to the input variable
            let mut matching_phrase = vec![pred_token.clone()];

            while input_depth < pred_depth {
                let pred_token = pred_token_iter.next().expect("pred_token");
                pred_depth += pred_token.open_depth;
                pred_depth -= pred_token.close_depth;

                matching_phrase.push(pred_token.clone());
            }

            let len = matching_phrase.len();
            if len == 1 {
                matching_phrase[0].open_depth = 0;
                matching_phrase[0].close_depth = 0;
            } else {
                matching_phrase[0].open_depth -= token.open_depth;
                matching_phrase[len - 1].close_depth -= token.close_depth;
            }

            let variable_already_matched = if let Some(&(_, ref existing_matches)) = result
                .iter()
                .chain(existing_matches.iter())
                .find(|&&(ref t, _)| *t == token.string)
            {
                if *existing_matches != matching_phrase {
                    // this match of the variable conflicted with an existing match
                    return None;
                }

                true
            } else {
                false
            };

            if !variable_already_matched {
                result.push((token.string, matching_phrase));
            }
        }

        pred_depth -= pred_token.close_depth;
        input_depth -= token.close_depth;
    }

    Some(result)
}

fn tokenize(string: &str, string_cache: &mut StringCache) -> Phrase {
    let mut string = format!("({})", string);

    lazy_static! {
        static ref RE1: Regex = Regex::new(r"\(\s*(\S+|`[^`]+`)\s*\)").unwrap();
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
        static ref RE2: Regex = Regex::new(r"`(.*?)`").unwrap();
    }

    let string1 = string.clone();
    let mut strings = RE2
        .captures_iter(&string1)
        .map(|c| c.get(1).expect("string_capture").as_str());

    string = RE2.replace_all(&string, "`").to_string();

    lazy_static! {
        static ref RE3: Regex = Regex::new(r"\(|\)|\s+|[^\(\)\s]+").unwrap();
    }

    let tokens = RE3
        .find_iter(&string)
        .map(|m| m.as_str())
        .filter(|s| !s.trim().is_empty())
        .collect::<Vec<_>>();

    let mut result = vec![];

    let mut open_depth = 0;
    let mut close_depth = 0;

    for (i, token) in tokens.iter().enumerate() {
        if *token == "(" {
            open_depth += 1;
            continue;
        }

        for t in tokens.iter().skip(i + 1) {
            if *t == ")" {
                close_depth += 1;
            } else {
                break;
            }
        }

        if *token == ")" {
            continue;
        }

        if *token == "`" {
            result.push(Token::new(
                strings.next().expect("string"),
                open_depth,
                close_depth,
                string_cache,
            ));
        } else {
            result.push(Token::new(token, open_depth, close_depth, string_cache));
        }
        open_depth = 0;
        close_depth = 0;
    }

    result
}

type FirstAtoms = Option<Atom>;
type FirstAtomsState = (usize, Atom);

fn extract_first_atoms_rule_input(phrase: &Phrase) -> FirstAtoms {
    if is_concrete_pred(phrase) {
        phrase
            .get(0)
            .option_filter(|t| !is_var_token(t))
            .map(|t| t.string)
    } else {
        None
    }
}

fn extract_first_atoms_state(state: &[Phrase]) -> Vec<FirstAtomsState> {
    let mut atoms: Vec<FirstAtomsState> = state
        .iter()
        .enumerate()
        .map(|(s_i, s)| extract_first_atoms_state_phrase(s_i, s))
        .collect();

    atoms.sort_unstable_by(|a, b| a.1.cmp(&b.1));

    atoms
}

fn extract_first_atoms_state_phrase(s_i: usize, phrase: &Phrase) -> FirstAtomsState {
    (s_i, phrase[0].string)
}

fn build_phrase(phrase: &Phrase, string_cache: &StringCache) -> String {
    let mut tokens = vec![];

    for t in phrase {
        let mut string = String::new();

        if let Some(s) = t.as_str(string_cache) {
            string += s;
        } else {
            string += &t.as_number().expect("number").to_string();
        }

        tokens.push(format!(
            "{}{}{}{}",
            String::from("(").repeat(t.open_depth as usize),
            if t.is_negated { "!" } else { "" },
            string,
            String::from(")").repeat(t.close_depth as usize)
        ));
    }

    tokens.join(" ")
}

fn print_state(state: &[Phrase], string_cache: &StringCache) {
    state
        .iter()
        .map(|p| build_phrase(p, string_cache))
        .for_each(|s| {
            println!("{}", s);
        });
}

fn rule_to_string(rule: &Rule, string_cache: &StringCache) -> String {
    let inputs = rule
        .inputs
        .iter()
        .map(|p| build_phrase(p, string_cache))
        .collect::<Vec<_>>()
        .join(" . ");

    let outputs = rule
        .outputs
        .iter()
        .map(|p| build_phrase(p, string_cache))
        .collect::<Vec<_>>()
        .join(" . ");

    format!("{:5}: {} = {}", rule.id, inputs, outputs)
}

fn test_rng() -> SmallRng {
    let seed = [
        123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123,
    ];

    SmallRng::from_seed(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule_new(inputs: Vec<Phrase>, outputs: Vec<Phrase>) -> Rule {
        Rule::new(0, inputs, outputs)
    }

    fn match_variables(input_tokens: &Phrase, pred_tokens: &Phrase) -> Option<Vec<Match>> {
        match_variables_with_existing(input_tokens, pred_tokens, &vec![])
    }

    #[test]
    fn context_from_text_state_test() {
        let mut context = Context::from_text(
            "at 0 0 wood . at 0 0 wood . #update \n\
             at 1 2 wood",
        );

        assert_eq!(
            context.state,
            [
                tokenize("at 0 0 wood", &mut context.string_cache),
                tokenize("at 0 0 wood", &mut context.string_cache),
                tokenize("#update", &mut context.string_cache),
                tokenize("at 1 2 wood", &mut context.string_cache),
            ]
        );
    }

    #[test]
    fn context_from_text_rules_test() {
        let mut context = Context::from_text(
            "at 0 0 wood . at 1 2 wood = at 1 0 wood \n\
             #test: \n\
             at 3 4 wood = at 5 6 wood . at 7 8 wood",
        );

        assert_eq!(
            context.rules,
            [
                Rule::new(
                    0,
                    vec![
                        tokenize("at 0 0 wood", &mut context.string_cache),
                        tokenize("at 1 2 wood", &mut context.string_cache),
                    ],
                    vec![tokenize("at 1 0 wood", &mut context.string_cache)]
                ),
                Rule::new(
                    1,
                    vec![
                        tokenize("#test", &mut context.string_cache),
                        tokenize("at 3 4 wood", &mut context.string_cache),
                    ],
                    vec![
                        tokenize("at 5 6 wood", &mut context.string_cache),
                        tokenize("at 7 8 wood", &mut context.string_cache),
                        tokenize("#test", &mut context.string_cache),
                    ],
                ),
            ]
        );
    }

    #[test]
    fn context_from_text_copy_test() {
        let mut context = Context::from_text("at 0 0 wood . $at 1 2 wood = at 1 0 wood");

        assert_eq!(
            context.rules,
            [Rule::new(
                0,
                vec![
                    tokenize("at 0 0 wood", &mut context.string_cache),
                    tokenize("at 1 2 wood", &mut context.string_cache),
                ],
                vec![
                    tokenize("at 1 2 wood", &mut context.string_cache),
                    tokenize("at 1 0 wood", &mut context.string_cache)
                ]
            ),]
        );
    }

    #[test]
    fn context_from_text_rules_newline_test() {
        let mut context = Context::from_text(
            "broken line 1 =\n\
             broken line 2 .\n\
             broken line 3 \n\
             . broken line 4\n\
             text\n\
             = `broken\ntext`",
        );

        assert_eq!(
            context.rules,
            [
                Rule::new(
                    0,
                    vec![tokenize("broken line 1", &mut context.string_cache)],
                    vec![
                        tokenize("broken line 2", &mut context.string_cache),
                        tokenize("broken line 3", &mut context.string_cache),
                        tokenize("broken line 4", &mut context.string_cache),
                    ],
                ),
                Rule::new(
                    1,
                    vec![tokenize("text", &mut context.string_cache)],
                    vec![tokenize("`broken\ntext`", &mut context.string_cache)],
                )
            ]
        );
    }

    #[test]
    fn context_from_text_comment_test() {
        let mut context = Context::from_text(
            "// comment 1\n\
             state 1\n\
             /* comment\n\
             2 */",
        );

        assert_eq!(
            context.state,
            [tokenize("state 1", &mut context.string_cache),]
        );
    }

    #[test]
    fn context_append_state_test() {
        let mut context = Context::from_text("test 1 2");

        context.append_state("test 3 4");

        assert_eq!(
            context.state,
            vec![
                tokenize("test 1 2", &mut context.string_cache),
                tokenize("test 3 4", &mut context.string_cache),
            ]
        );
    }

    #[test]
    fn context_find_matching_rules_test() {
        let mut context = Context::from_text(
            "test 1 2 . test 3 4 . test 5 6\n\
             \n\
             test 1 2 . test 5 6 = match\n\
             test 1 2 . nomatch = nomatch\n\
             test 3 4 . test 5 6 = match",
        );

        assert_eq!(
            context.find_matching_rules(),
            [
                Rule::new(
                    0,
                    vec![
                        tokenize("test 1 2", &mut context.string_cache),
                        tokenize("test 5 6", &mut context.string_cache),
                    ],
                    vec![tokenize("match", &mut context.string_cache)]
                ),
                Rule::new(
                    2,
                    vec![
                        tokenize("test 3 4", &mut context.string_cache),
                        tokenize("test 5 6", &mut context.string_cache),
                    ],
                    vec![tokenize("match", &mut context.string_cache)],
                ),
            ]
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
        ).with_test_rng();

        update(&mut context, |_: &Phrase| None);

        assert_eq!(
            context.state,
            [
                tokenize("at 1 1 wood", &mut context.string_cache),
                tokenize("at 0 0 wood", &mut context.string_cache),
                tokenize("at 0 1 fire", &mut context.string_cache),
                tokenize("at 1 1 fire", &mut context.string_cache),
                tokenize("at 0 0 fire", &mut context.string_cache),
            ]
        );
    }

    #[test]
    fn token_test() {
        let mut string_cache = StringCache::new();
        assert!(!is_var_token(&Token::new("tt1", 1, 1, &mut string_cache)));
        assert!(!is_var_token(&Token::new("tT1", 1, 1, &mut string_cache)));
        assert!(!is_var_token(&Token::new("1", 1, 1, &mut string_cache)));
        assert!(!is_var_token(&Token::new("1Tt", 1, 1, &mut string_cache)));
        assert!(is_var_token(&Token::new("T", 1, 1, &mut string_cache)));
        assert!(is_var_token(&Token::new("TT1", 1, 1, &mut string_cache)));
        assert!(is_var_token(&Token::new("TT1'", 1, 1, &mut string_cache)));
    }

    #[test]
    fn rule_matches_state_truthiness_test() {
        let mut string_cache = StringCache::new();

        let mut test_cases = vec![
            (
                rule_new(
                    vec![
                        tokenize("t1 t3 t2", &mut string_cache),
                        tokenize("t1 t2 t3", &mut string_cache),
                    ],
                    vec![],
                ),
                vec![
                    tokenize("t1 t3 t2", &mut string_cache),
                    tokenize("t1 t2 t3", &mut string_cache),
                ],
                true,
            ),
            (
                rule_new(vec![tokenize("t1 T2 T3", &mut string_cache)], vec![]),
                vec![tokenize("t1 t2 t3", &mut string_cache)],
                true,
            ),
            (
                rule_new(
                    vec![
                        tokenize("t1 T3 T2", &mut string_cache),
                        tokenize("t1 T2 T3", &mut string_cache),
                    ],
                    vec![],
                ),
                vec![
                    tokenize("t1 t2 t3", &mut string_cache),
                    tokenize("t1 t2 t3", &mut string_cache),
                ],
                false,
            ),
            (
                rule_new(
                    vec![
                        tokenize("t1 T3 T2", &mut string_cache),
                        tokenize("t1 T2 T3", &mut string_cache),
                    ],
                    vec![],
                ),
                vec![
                    tokenize("t1 t3 t2", &mut string_cache),
                    tokenize("t1 t2 t3", &mut string_cache),
                ],
                true,
            ),
            (
                rule_new(
                    vec![
                        tokenize("t1 T1 T2", &mut string_cache),
                        tokenize("+ T1 T2 T2", &mut string_cache),
                    ],
                    vec![],
                ),
                vec![tokenize("t1 1 2", &mut string_cache)],
                false,
            ),
            (
                rule_new(
                    vec![
                        tokenize("t1 T1 T2", &mut string_cache),
                        tokenize("+ T1 T2 T2", &mut string_cache),
                    ],
                    vec![],
                ),
                vec![tokenize("t1 0 2", &mut string_cache)],
                true,
            ),
            (
                rule_new(
                    vec![
                        tokenize("t1 T1 T2", &mut string_cache),
                        tokenize("t1 T1 T2", &mut string_cache),
                    ],
                    vec![],
                ),
                vec![tokenize("t1 0 2", &mut string_cache)],
                false,
            ),
            // successful match with backwards predicates at first and last position
            (
                rule_new(
                    vec![
                        tokenize("+ T1 T2 T2", &mut string_cache),
                        tokenize("t1 T1 T2", &mut string_cache),
                        tokenize("t3 T3 T4", &mut string_cache),
                        tokenize("+ T3 T4 T2", &mut string_cache),
                    ],
                    vec![],
                ),
                vec![
                    tokenize("t1 0 2", &mut string_cache),
                    tokenize("t3 -2 4", &mut string_cache),
                ],
                true,
            ),
            // unsuccessful match with backwards predicates at first and last position
            (
                rule_new(
                    vec![
                        tokenize("+ T1 T2 T2", &mut string_cache),
                        tokenize("t1 T1 T2", &mut string_cache),
                        tokenize("t3 T3 T4", &mut string_cache),
                        tokenize("+ T3 T4 0", &mut string_cache),
                    ],
                    vec![],
                ),
                vec![
                    tokenize("t1 0 2", &mut string_cache),
                    tokenize("t3 -2 4", &mut string_cache),
                ],
                false,
            ),
            (
                rule_new(
                    vec![
                        tokenize("+ T1 1 T2", &mut string_cache),
                        tokenize("+ T3 1 T4", &mut string_cache),
                        tokenize("t1 T1 T4", &mut string_cache),
                    ],
                    vec![],
                ),
                vec![tokenize("t1 0 1", &mut string_cache)],
                true,
            ),
            (
                rule_new(
                    vec![
                        tokenize("first", &mut string_cache),
                        // failing backwards predicate
                        tokenize("+ 3 4 5", &mut string_cache),
                        tokenize("at X Y fire", &mut string_cache),
                    ],
                    vec![],
                ),
                vec![
                    tokenize("at 1 1 fire", &mut string_cache),
                    tokenize("at 1 -1 fire", &mut string_cache),
                    tokenize("first", &mut string_cache),
                ],
                false,
            ),
            (
                rule_new(
                    vec![
                        tokenize("at X Y fire", &mut string_cache),
                        tokenize("< 0 X", &mut string_cache),
                    ],
                    vec![],
                ),
                vec![
                    tokenize("at 0 0 fire", &mut string_cache),
                    tokenize("at 2 0 fire", &mut string_cache),
                ],
                true,
            ),
        ];

        for (rule, state, expected) in test_cases.drain(..) {
            let result = rule_matches_state(
                &rule,
                &state,
                &mut |_: &Phrase| None,
                &mut string_cache,
                &extract_first_atoms_state(&state),
            );

            if expected {
                assert!(result.is_some());
            } else {
                assert!(result.is_none());
            }
        }
    }

    #[test]
    fn rule_matches_state_truthiness_negated_test() {
        let mut string_cache = StringCache::new();

        let mut test_cases = vec![
            (
                rule_new(vec![tokenize("!test", &mut string_cache)], vec![]),
                vec![
                    tokenize("foo", &mut string_cache),
                    tokenize("bar", &mut string_cache),
                ],
                true,
            ),
            (
                rule_new(vec![tokenize("!test", &mut string_cache)], vec![]),
                vec![
                    tokenize("foo", &mut string_cache),
                    tokenize("test", &mut string_cache),
                    tokenize("bar", &mut string_cache),
                ],
                false,
            ),
            (
                rule_new(
                    vec![
                        tokenize("test", &mut string_cache),
                        tokenize("!test", &mut string_cache),
                    ],
                    vec![],
                ),
                vec![
                    tokenize("foo", &mut string_cache),
                    tokenize("test", &mut string_cache),
                    tokenize("bar", &mut string_cache),
                ],
                true,
            ),
            (
                rule_new(
                    vec![
                        tokenize("test", &mut string_cache),
                        tokenize("!test", &mut string_cache),
                    ],
                    vec![],
                ),
                vec![
                    tokenize("foo", &mut string_cache),
                    tokenize("test", &mut string_cache),
                    tokenize("test", &mut string_cache),
                    tokenize("bar", &mut string_cache),
                ],
                false,
            ),
            (
                rule_new(
                    vec![
                        tokenize("!test A B", &mut string_cache),
                        tokenize("foo A B", &mut string_cache),
                    ],
                    vec![],
                ),
                vec![
                    tokenize("foo 1 2", &mut string_cache),
                    tokenize("test 1 3", &mut string_cache),
                ],
                true,
            ),
            (
                rule_new(
                    vec![
                        tokenize("!test A B", &mut string_cache),
                        tokenize("foo A B", &mut string_cache),
                    ],
                    vec![],
                ),
                vec![
                    tokenize("foo 1 2", &mut string_cache),
                    tokenize("test 1 2", &mut string_cache),
                ],
                false,
            ),
        ];

        for (rule, state, expected) in test_cases.drain(..) {
            let result = rule_matches_state(
                &rule,
                &state,
                &mut |_: &Phrase| None,
                &mut string_cache,
                &extract_first_atoms_state(&state),
            );

            if expected {
                assert!(result.is_some());
            } else {
                assert!(result.is_none());
            }
        }
    }

    #[test]
    fn rule_matches_state_output_test() {
        let mut string_cache = StringCache::new();

        let mut test_cases = vec![
            (
                rule_new(
                    vec![
                        tokenize("t1 T1 T2", &mut string_cache),
                        tokenize("+ T1 T2 T3'", &mut string_cache),
                        tokenize("+ T1 T4' T2", &mut string_cache),
                    ],
                    vec![
                        tokenize("t3 T3'", &mut string_cache),
                        tokenize("t4 T4'", &mut string_cache),
                    ],
                ),
                vec![tokenize("t1 3 4", &mut string_cache)],
                rule_new(
                    vec![tokenize("t1 3 4", &mut string_cache)],
                    vec![
                        tokenize("t3 7", &mut string_cache),
                        tokenize("t4 1", &mut string_cache),
                    ],
                ),
            ),
            (
                rule_new(
                    vec![
                        tokenize("#collision", &mut string_cache),
                        tokenize("block-falling ID X Y", &mut string_cache),
                        tokenize("+ Y1 1 Y", &mut string_cache),
                        tokenize("block-set ID2 X Y1", &mut string_cache),
                    ],
                    vec![
                        tokenize("block-setting ID X Y", &mut string_cache),
                        tokenize("#collision", &mut string_cache),
                        tokenize("block-set ID2 X Y1", &mut string_cache),
                    ],
                ),
                vec![
                    tokenize("block-set 0 5 1", &mut string_cache),
                    tokenize("block-set 1 6 1", &mut string_cache),
                    tokenize("block-set 3 6 0", &mut string_cache),
                    tokenize("block-falling 7 6 2", &mut string_cache),
                    tokenize("#collision", &mut string_cache),
                    tokenize("block-falling 6 5 2", &mut string_cache),
                    tokenize("block-set 2 5 0", &mut string_cache),
                ],
                rule_new(
                    vec![
                        tokenize("#collision", &mut string_cache),
                        tokenize("block-falling 7 6 2", &mut string_cache),
                        tokenize("block-set 1 6 1", &mut string_cache),
                    ],
                    vec![
                        tokenize("block-setting 7 6 2", &mut string_cache),
                        tokenize("#collision", &mut string_cache),
                        tokenize("block-set 1 6 1", &mut string_cache),
                    ],
                ),
            ),
        ];

        for (rule, state, expected) in test_cases.drain(..) {
            let result = rule_matches_state(
                &rule,
                &state,
                &mut |_: &Phrase| None,
                &mut string_cache,
                &extract_first_atoms_state(&state),
            );

            assert!(result.is_some());
            assert_eq!(result.unwrap(), expected);
        }
    }

    #[test]
    fn evaluate_backwards_pred_test() {
        let mut string_cache = StringCache::new();

        let mut test_cases = vec![
            (
                tokenize("+ A 2 3", &mut string_cache),
                Some(tokenize("+ 1 2 3", &mut string_cache)),
            ),
            (
                tokenize("+ 1 B 3", &mut string_cache),
                Some(tokenize("+ 1 2 3", &mut string_cache)),
            ),
            (
                tokenize("+ 1 2 C", &mut string_cache),
                Some(tokenize("+ 1 2 3", &mut string_cache)),
            ),
            (tokenize("+ 1 2 4", &mut string_cache), None),
        ];

        for (input, expected) in test_cases.drain(..) {
            assert_eq!(evaluate_backwards_pred(&input, &mut string_cache), expected);
        }
    }

    #[test]
    fn assign_vars_test() {
        let mut string_cache = StringCache::new();

        let mut test_cases = vec![
            (
                tokenize("+ T1 T2 T3", &mut string_cache),
                vec![
                    (
                        string_cache.str_to_atom("T1"),
                        tokenize("1", &mut string_cache),
                    ),
                    (
                        string_cache.str_to_atom("T2"),
                        tokenize("2", &mut string_cache),
                    ),
                ],
                tokenize("+ 1 2 T3", &mut string_cache),
            ),
            (
                tokenize("T1 (T2 T3)", &mut string_cache),
                vec![
                    (
                        string_cache.str_to_atom("T1"),
                        tokenize("t11 t12", &mut string_cache),
                    ),
                    (
                        string_cache.str_to_atom("T3"),
                        tokenize("t31 (t32 t33)", &mut string_cache),
                    ),
                ],
                tokenize("(t11 t12) (T2 (t31 (t32 t33)))", &mut string_cache),
            ),
            (
                tokenize("T1 !T2", &mut string_cache),
                vec![(
                    string_cache.str_to_atom("T2"),
                    tokenize("t11 t12", &mut string_cache),
                )],
                tokenize("T1 (!t11 t12)", &mut string_cache),
            ),
        ];

        for (tokens, matches, expected) in test_cases.drain(..) {
            assert_eq!(assign_vars(&tokens, &matches), expected);
        }
    }

    #[test]
    fn match_variables_test() {
        let mut string_cache = StringCache::new();

        let mut test_cases = vec![
            (
                tokenize("t1 T2 T3", &mut string_cache),
                tokenize("t1 t2 t3", &mut string_cache),
                Some(vec![
                    (
                        string_cache.str_to_atom("T2"),
                        tokenize("t2", &mut string_cache),
                    ),
                    (
                        string_cache.str_to_atom("T3"),
                        tokenize("t3", &mut string_cache),
                    ),
                ]),
            ),
            (
                tokenize("t1 T2", &mut string_cache),
                tokenize("t1 (t21 t22)", &mut string_cache),
                Some(vec![(
                    string_cache.str_to_atom("T2"),
                    tokenize("t21 t22", &mut string_cache),
                )]),
            ),
            (
                tokenize("t1 (t21 T22 t23) T3", &mut string_cache),
                tokenize("t1 (t21 (t221 t222 t223) t23) t3", &mut string_cache),
                Some(vec![
                    (
                        string_cache.str_to_atom("T22"),
                        tokenize("t221 t222 t223", &mut string_cache),
                    ),
                    (
                        string_cache.str_to_atom("T3"),
                        tokenize("t3", &mut string_cache),
                    ),
                ]),
            ),
            (
                tokenize("t1 T2 T3", &mut string_cache),
                tokenize("t1 t2 (t3 t2)", &mut string_cache),
                Some(vec![
                    (
                        string_cache.str_to_atom("T2"),
                        tokenize("t2", &mut string_cache),
                    ),
                    (
                        string_cache.str_to_atom("T3"),
                        tokenize("t3 t2", &mut string_cache),
                    ),
                ]),
            ),
            (
                tokenize("t1 T2", &mut string_cache),
                tokenize("t1 (t2 t3 (t3 t2))", &mut string_cache),
                Some(vec![(
                    string_cache.str_to_atom("T2"),
                    tokenize("t2 t3 (t3 t2)", &mut string_cache),
                )]),
            ),
            (
                tokenize("t1 T2", &mut string_cache),
                tokenize("t1 ((t2 t3) t3 t2)", &mut string_cache),
                Some(vec![(
                    string_cache.str_to_atom("T2"),
                    tokenize("(t2 t3) t3 t2", &mut string_cache),
                )]),
            ),
            (
                tokenize("t1 (t2 t3 T2)", &mut string_cache),
                tokenize("t1 (t2 t3 (t3 t2))", &mut string_cache),
                Some(vec![(
                    string_cache.str_to_atom("T2"),
                    tokenize("t3 t2", &mut string_cache),
                )]),
            ),
            (
                tokenize("t1 (T2 t3 t2)", &mut string_cache),
                tokenize("t1 ((t2 t3) t3 t2)", &mut string_cache),
                Some(vec![(
                    string_cache.str_to_atom("T2"),
                    tokenize("t2 t3", &mut string_cache),
                )]),
            ),
            (
                tokenize("t1 T2 T3", &mut string_cache),
                tokenize("t1 (t2 t3) (t3 t2)", &mut string_cache),
                Some(vec![
                    (
                        string_cache.str_to_atom("T2"),
                        tokenize("t2 t3", &mut string_cache),
                    ),
                    (
                        string_cache.str_to_atom("T3"),
                        tokenize("t3 t2", &mut string_cache),
                    ),
                ]),
            ),
            (
                tokenize("t1 t3", &mut string_cache),
                tokenize("t1 t3", &mut string_cache),
                Some(vec![]),
            ),
            (
                tokenize("t1 t3", &mut string_cache),
                tokenize("t1 (t21 t23)", &mut string_cache),
                None,
            ),
            (
                tokenize("t1 T3", &mut string_cache),
                tokenize("t1 t2 t3", &mut string_cache),
                None,
            ),
            (
                tokenize("t1 T3 t3", &mut string_cache),
                tokenize("t1 t2", &mut string_cache),
                None,
            ),
            (
                tokenize("t1 T3 T3", &mut string_cache),
                tokenize("t1 t2 t3", &mut string_cache),
                None,
            ),
            (
                tokenize("t1 T3 T3", &mut string_cache),
                tokenize("t1 t3 t3", &mut string_cache),
                Some(vec![(
                    string_cache.str_to_atom("T3"),
                    tokenize("t3", &mut string_cache),
                )]),
            ),
        ];

        for (input_tokens, pred_tokens, expected) in test_cases.drain(..) {
            assert_eq!(match_variables(&input_tokens, &pred_tokens), expected);
        }
    }

    #[test]
    fn match_variables_with_existing_test() {
        let mut string_cache = StringCache::new();

        let input_tokens = tokenize("T1 T2 T3", &mut string_cache);
        let pred_tokens = tokenize("t1 t2 t3", &mut string_cache);
        let existing_matches = vec![(
            string_cache.str_to_atom("T2"),
            tokenize("t2", &mut string_cache),
        )];

        let result = match_variables_with_existing(&input_tokens, &pred_tokens, &existing_matches);

        assert_eq!(
            result,
            Some(vec![
                (
                    string_cache.str_to_atom("T1"),
                    tokenize("t1", &mut string_cache),
                ),
                (
                    string_cache.str_to_atom("T3"),
                    tokenize("t3", &mut string_cache),
                ),
            ])
        )
    }

    #[test]
    fn tokenize_test() {
        let mut string_cache = StringCache::new();

        assert_eq!(
            tokenize("t1", &mut string_cache),
            [Token::new("t1", 0, 0, &mut string_cache)]
        );

        assert_eq!(
            tokenize("t1 (t21 (t221 t222 t223) t23) t3", &mut string_cache),
            [
                Token::new("t1", 1, 0, &mut string_cache),
                Token::new("t21", 1, 0, &mut string_cache),
                Token::new("t221", 1, 0, &mut string_cache),
                Token::new("t222", 0, 0, &mut string_cache),
                Token::new("t223", 0, 1, &mut string_cache),
                Token::new("t23", 0, 1, &mut string_cache),
                Token::new("t3", 0, 1, &mut string_cache),
            ]
        );

        assert_eq!(
            tokenize("t1 t2 (((t3 )) t4)", &mut string_cache),
            [
                Token::new("t1", 1, 0, &mut string_cache),
                Token::new("t2", 0, 0, &mut string_cache),
                Token::new("t3", 1, 0, &mut string_cache),
                Token::new("t4", 0, 2, &mut string_cache),
            ]
        );

        assert_eq!(
            tokenize("(t1 t2) (t3 t4)", &mut string_cache),
            [
                Token::new("t1", 2, 0, &mut string_cache),
                Token::new("t2", 0, 1, &mut string_cache),
                Token::new("t3", 1, 0, &mut string_cache),
                Token::new("t4", 0, 2, &mut string_cache),
            ]
        );

        assert_eq!(
            tokenize("t1 t2 (t3'' t4')", &mut string_cache),
            [
                Token::new("t1", 1, 0, &mut string_cache),
                Token::new("t2", 0, 0, &mut string_cache),
                Token::new("t3''", 1, 0, &mut string_cache),
                Token::new("t4'", 0, 2, &mut string_cache),
            ]
        );
    }

    #[test]
    fn tokenize_string_test() {
        let mut string_cache = StringCache::new();

        assert_eq!(
            tokenize("`string here`", &mut string_cache),
            [Token::new("string here", 0, 0, &mut string_cache),]
        );

        assert_eq!(
            tokenize("`one string` `two strings`", &mut string_cache),
            [
                Token::new("one string", 1, 0, &mut string_cache),
                Token::new("two strings", 0, 1, &mut string_cache),
            ]
        );

        assert_eq!(
            tokenize(
                "t1 t2 (((`string here` )) `final string`)",
                &mut string_cache
            ),
            [
                Token::new("t1", 1, 0, &mut string_cache),
                Token::new("t2", 0, 0, &mut string_cache),
                Token::new("string here", 1, 0, &mut string_cache),
                Token::new("final string", 0, 2, &mut string_cache),
            ]
        );
    }
}
