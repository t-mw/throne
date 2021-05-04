use crate::matching::*;
use crate::rule::{LineColSpan, Rule};
use crate::string_cache::{Atom, StringCache};
use crate::token::*;

use pest::iterators::Pair;
use pest::Parser;
use rand::rngs::SmallRng;
use rand::Rng;

use std::collections::HashMap;
use std::fmt;

pub struct ParseResult {
    pub rules: Vec<Rule>,
    pub state: Vec<Vec<Token>>,
}

#[derive(Debug)]
pub struct Error {
    pub pest: pest::error::Error<generated::Rule>,
}

impl From<pest::error::Error<generated::Rule>> for Error {
    fn from(pest: pest::error::Error<generated::Rule>) -> Self {
        Error { pest }
    }
}

impl std::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.pest)
    }
}

mod generated {
    #[derive(Parser)]
    #[grammar = "throne.pest"]
    pub struct Parser;
}

pub fn parse(
    text: &str,
    mut string_cache: &mut StringCache,
    rng: &mut SmallRng,
) -> Result<ParseResult, Error> {
    let text = text.replace("()", "qui");

    let file = generated::Parser::parse(generated::Rule::file, &text)?
        .next()
        .unwrap();

    let mut state: Vec<Vec<Token>> = vec![];
    let mut rules: Vec<(Rule, bool)> = vec![];
    let mut backwards_preds: Vec<(VecPhrase, Vec<VecPhrase>)> = vec![];
    let mut enable_unused_warnings = true;

    for line in file.into_inner() {
        match line.as_rule() {
            generated::Rule::prefixed => {
                let mut prefixed = line.into_inner();

                let prefix_inputs_pair = prefixed.next().unwrap();
                for pair in prefixed {
                    let r = pair_to_throne_rule(
                        pair,
                        Some(prefix_inputs_pair.clone()),
                        &mut string_cache,
                        enable_unused_warnings,
                    );
                    rules.push((r, enable_unused_warnings));
                }
            }
            generated::Rule::backwards_def => {
                check_rule_variables(line.clone(), enable_unused_warnings);

                let mut backwards_def = line.into_inner();

                let mut first_phrase =
                    tokenize(backwards_def.next().unwrap().as_str(), &mut string_cache);

                let mut var_replacements =
                    replace_variables(&mut first_phrase, &mut string_cache, None, rng);

                let mut other_phrases = backwards_def
                    .map(|phrase| tokenize(phrase.as_str(), &mut string_cache))
                    .collect::<Vec<_>>();

                for phrase in &mut other_phrases {
                    var_replacements =
                        replace_variables(phrase, &mut string_cache, Some(var_replacements), rng);
                }

                backwards_preds.push((first_phrase, other_phrases));
            }
            generated::Rule::rule => {
                rules.push((
                    pair_to_throne_rule(line, None, &mut string_cache, enable_unused_warnings),
                    enable_unused_warnings,
                ));
            }
            generated::Rule::state => {
                for phrase in line.into_inner() {
                    state.push(tokenize(phrase.as_str(), &mut string_cache));
                }
            }
            generated::Rule::compiler_disable_unused_warnings => {
                enable_unused_warnings = false;
            }
            generated::Rule::compiler_enable_unused_warnings => {
                enable_unused_warnings = true;
            }
            generated::Rule::EOI => (),
            _ => unreachable!("{}", line),
        }
    }

    let mut new_rules = vec![];
    for (rule, enable_unused_warnings_for_rule) in &rules {
        if let Some(mut replaced_rules) = replace_backwards_preds(
            &rule,
            &backwards_preds,
            &string_cache,
            *enable_unused_warnings_for_rule,
        ) {
            new_rules.append(&mut replaced_rules);
        }
    }

    for (i, rule) in new_rules.iter_mut().enumerate() {
        rule.id = i as i32;
    }

    Ok(ParseResult {
        rules: new_rules,
        state,
    })
}

fn pair_to_throne_rule(
    rule_pair: Pair<generated::Rule>,
    prefix_inputs_pair: Option<Pair<generated::Rule>>,
    string_cache: &mut StringCache,
    enable_unused_warnings: bool,
) -> Rule {
    check_rule_variables(rule_pair.clone(), enable_unused_warnings);

    let source_span: LineColSpan = rule_pair.as_span().into();

    let mut pairs = rule_pair.into_inner();
    let inputs_pair = pairs.next().unwrap();
    let outputs_pair = pairs.next().unwrap();

    let mut inputs = vec![];
    let mut outputs = vec![];

    let input_pairs = inputs_pair.into_inner();
    let has_input_qui = input_pairs.clone().any(|p| {
        let input_phrase = p.into_inner().next().unwrap();
        input_phrase.as_rule() == generated::Rule::qui
    });

    let mut handle_input_phrase = |input_phrase: Pair<generated::Rule>, is_prefix_input: bool| {
        match input_phrase.as_rule() {
            generated::Rule::copy_phrase => {
                let copy_phrase = tokenize(
                    input_phrase.into_inner().next().unwrap().as_str(),
                    string_cache,
                );

                inputs.push(copy_phrase.clone());
                outputs.push(copy_phrase);
            }
            // stage phrases have the special behavior of acting as copy phrases when used as
            // prefixes, except when the prefixed rule includes a qui.
            generated::Rule::stage_phrase => {
                let stage_phrase = tokenize(input_phrase.as_str(), string_cache);
                if !has_input_qui && is_prefix_input {
                    outputs.push(stage_phrase.clone());
                }
                inputs.push(stage_phrase);
            }
            _ => {
                inputs.push(tokenize(input_phrase.as_str(), string_cache));
            }
        }
    };

    if let Some(prefix_inputs_pair) = prefix_inputs_pair {
        // insert stages at beginning of rule input, so that 'first atoms' optimization is effective
        let prefix_input_pairs = prefix_inputs_pair.into_inner();
        for p in prefix_input_pairs.clone() {
            let input_phrase = p.into_inner().next().unwrap();
            if input_phrase.as_rule() == generated::Rule::stage_phrase {
                handle_input_phrase(input_phrase, true);
            }
        }

        for p in prefix_input_pairs {
            let input_phrase = p.into_inner().next().unwrap();
            if input_phrase.as_rule() != generated::Rule::stage_phrase {
                handle_input_phrase(input_phrase, true);
            }
        }
    }

    for p in input_pairs {
        let input_phrase = p.into_inner().next().unwrap();
        handle_input_phrase(input_phrase, false);
    }

    for p in outputs_pair.into_inner() {
        let output_phrase = p.into_inner().next().unwrap();

        match output_phrase.as_rule() {
            generated::Rule::qui => (),
            _ => outputs.push(tokenize(output_phrase.as_str(), string_cache)),
        }
    }

    Rule::new(0, inputs, outputs, source_span)
}

// for each backwards predicate, replace it with the corresponding phrase
fn replace_backwards_preds(
    rule: &Rule,
    backwards_preds: &Vec<(VecPhrase, Vec<VecPhrase>)>,
    string_cache: &StringCache,
    enable_unused_warnings: bool,
) -> Option<Vec<Rule>> {
    let mut backwards_preds_per_input = vec![vec![]; rule.inputs.len()];
    let mut backwards_pred_pointers = vec![0; rule.inputs.len()];

    for (i_i, input) in rule.inputs.iter().enumerate() {
        if input[0].flag == TokenFlag::BackwardsPred(BackwardsPred::Custom) {
            let mut matched = false;

            for (b_i, (first_phrase, _)) in backwards_preds.iter().enumerate() {
                if match_variables_twoway(input, first_phrase, &mut vec![]) {
                    backwards_preds_per_input[i_i].push(b_i);
                    matched = true;
                }
            }

            if !matched {
                if enable_unused_warnings {
                    println!("WARNING: backwards predicate in rule did not match '{}'. Check that the backwards predicate is defined.", rule.to_string(string_cache));
                }
                return None;
            }
        }
    }

    let first = backwards_preds_per_input.iter().position(|v| v.len() > 0);
    let last = backwards_preds_per_input
        .iter()
        .rev()
        .position(|v| v.len() > 0)
        .map(|idx| backwards_preds_per_input.len() - 1 - idx);

    let backwards_pred_input_range = match (first, last) {
        (Some(first), Some(last)) => (first, last),
        // no backwards predicates in rule
        _ => return Some(vec![rule.clone()]),
    };

    let mut new_rules = vec![];
    'outer: loop {
        let mut nonvariable_matches: Vec<Match> = vec![];
        let mut matches: Vec<Match> = vec![];

        let mut matched = true;

        for (i_i, input) in rule.inputs.iter().enumerate() {
            let backwards_preds_for_input = &backwards_preds_per_input[i_i];
            if backwards_preds_for_input.len() > 0 {
                let backwards_pred_pointer = backwards_pred_pointers[i_i];
                let (first_phrase, _) =
                    &backwards_preds[backwards_preds_for_input[backwards_pred_pointer]];

                // Match variables from rule input b.p. to b.p. definition first phrase,
                // ignoring matches that have already been made, unless those matches
                // were to non-variables in the backwards predicate definition.
                //
                // i.e. Given <<back1 A B . <<back2 A B = ...
                // we only need to check that the variables between the
                // backwards predicates are compatible, if they match
                // non-variable atoms in the backwards predicate definition.
                if !match_variables_twoway(input, first_phrase, &mut nonvariable_matches) {
                    matched = false;
                    break;
                }

                for m in &nonvariable_matches {
                    if matches
                        .iter()
                        .find(|other_m| other_m.atom == m.atom)
                        .is_none()
                    {
                        matches.push(m.clone());
                    }
                }

                nonvariable_matches = nonvariable_matches
                    .iter()
                    .filter(|m| !is_var_token(&m.phrase[0]))
                    .cloned()
                    .collect::<Vec<_>>();
            }
        }

        if matched {
            let mut new_inputs = vec![];

            for (i_i, input) in rule.inputs.iter().enumerate() {
                let backwards_preds_for_input = &backwards_preds_per_input[i_i];
                if backwards_preds_for_input.len() > 0 {
                    let backwards_pred_pointer = backwards_pred_pointers[i_i];
                    let (first_phrase, other_phrases) =
                        &backwards_preds[backwards_preds_for_input[backwards_pred_pointer]];

                    let complete_input_phrase = assign_vars(input, &nonvariable_matches);
                    let mut inverse_matches = vec![];

                    match_variables_assuming_compatible_structure(
                        &first_phrase,
                        &complete_input_phrase,
                        &mut inverse_matches,
                    );

                    for phrase in other_phrases {
                        // replace variable names in backwards predicate phrase
                        // with variable names from original rule
                        let complete_phrase = assign_vars(phrase, &inverse_matches);
                        new_inputs.push(complete_phrase);
                    }
                } else {
                    new_inputs.push(assign_vars(input, &nonvariable_matches));
                }
            }

            let mut new_rule = rule.clone();

            new_rule.inputs = new_inputs;
            new_rule.outputs = rule
                .outputs
                .iter()
                .map(|output| assign_vars(output, &nonvariable_matches))
                .collect();
            new_rules.push(new_rule);
        }

        // find next permutation of backwards predicates in rule
        for i in (backwards_pred_input_range.0..=backwards_pred_input_range.1).rev() {
            let backwards_preds_for_input = &backwards_preds_per_input[i];

            if backwards_preds_for_input.len() == 0 {
                continue;
            }

            let at_end = backwards_pred_pointers[i] == backwards_preds_for_input.len() - 1;
            if at_end && i == backwards_pred_input_range.0 {
                // finished checking all permutations
                break 'outer;
            }

            if at_end {
                backwards_pred_pointers[i] = 0;
            } else {
                backwards_pred_pointers[i] += 1;
                break;
            }
        }
    }

    Some(new_rules)
}

// replace all variable tokens in a phrase with unique tokens,
// optionally using any existing mappings provided
fn replace_variables(
    phrase: &mut Vec<Token>,
    string_cache: &mut StringCache,
    existing_map: Option<HashMap<Atom, Atom>>,
    rng: &mut SmallRng,
) -> HashMap<Atom, Atom> {
    let mut existing_map = existing_map.unwrap_or(HashMap::new());

    for token in phrase {
        if token.flag != TokenFlag::Variable {
            continue;
        }

        if let Some(replacement) = existing_map.get(&token.atom) {
            token.atom = *replacement;
        } else {
            loop {
                let s = string_cache.atom_to_str(token.atom).unwrap();
                let replacement_s = format!("{}_BACK{}{}", s, rng.gen::<u32>(), rng.gen::<u32>());
                let replacement = string_cache.str_to_atom(&replacement_s);

                if existing_map.contains_key(&replacement) {
                    continue;
                }

                existing_map.insert(token.atom, replacement);
                token.atom = replacement;
                break;
            }
        }
    }

    existing_map
}

fn check_rule_variables(pair: Pair<generated::Rule>, enable_unused_warnings: bool) {
    if !enable_unused_warnings {
        return;
    }

    let rule_str = pair.as_str();
    let inner = pair.into_inner();
    let mut var_counts = HashMap::new();

    for p in inner.flatten() {
        if let generated::Rule::atom_var = p.as_rule() {
            let count = var_counts.entry(p.as_str()).or_insert(0);
            *count += 1;
        }
    }

    for (var_name, count) in &var_counts {
        if *count == 1 {
            println!("WARNING: {} was only used once in '{}'. Check for errors or replace with a wildcard.", var_name, rule_str);
        }
    }
}

impl From<pest::Span<'_>> for LineColSpan {
    fn from(span: pest::Span) -> Self {
        let start_line_col = span.start_pos().line_col();
        let end_line_col = span.end_pos().line_col();
        LineColSpan {
            line_start: start_line_col.0,
            line_end: end_line_col.0,
            col_start: start_line_col.1,
            col_end: end_line_col.1,
        }
    }
}
